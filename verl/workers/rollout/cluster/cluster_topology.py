"""
ClusterTopology — 集群拓扑与健康管理层。

职责（仅此而已）:
  - 维护 ServerEndpoint 列表及其健康状态
  - 后台线程定期探测 /health + /pool_status，更新 is_healthy / pending_count
  - 提供 healthy_endpoints() 和 pending_of(ep) 供 ClusterDispatcher 做路由决策
  - 启动时等待所有 server 就绪

不做:
  - HTTP 提交 / 拉取（在 ServerEndpoint）
  - 路由决策（在 ClusterDispatcher）
  - FSDP / NCCL（在 RolloutClusterClient）
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .server_endpoint import ServerEndpoint

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 5    # s，单次健康检查的网络超时


class ClusterTopology:
    """
    管理一批 ServerEndpoint 的生命周期与健康状态。

    public API:
        healthy_endpoints()       → list[ServerEndpoint]
        pending_of(ep)            → int   最近一次探测缓存的 pending 数
        wait_until_ready(timeout) → None  阻塞，启动时调用一次
        start_health_watcher()    → None
        shutdown()                → None
    """

    def __init__(
        self,
        server_urls: list[str],
        health_check_interval: float = 30.0
    ):
        if not server_urls:
            raise ValueError("server_urls must not be empty")

        self.endpoints: list[ServerEndpoint] = [
            ServerEndpoint(url=url, index=i)
            for i, url in enumerate(server_urls)
        ]
        self.health_check_interval = health_check_interval

        self._stop      = threading.Event()
        self._watcher: Optional[threading.Thread] = None

        # 并行健康探测线程池
        self._probe_pool = ThreadPoolExecutor(
            max_workers=len(self.endpoints),
            thread_name_prefix="topology-probe",
        )

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def healthy_endpoints(self) -> list[ServerEndpoint]:
        """返回当前标记为健康的 endpoint 列表（快照，无锁读）。"""
        return [ep for ep in self.endpoints if ep.is_healthy]

    def pending_of(self, ep: ServerEndpoint) -> int:
        """
        返回 ep 最近一次健康探测时缓存的 pending_count。
        供 ClusterDispatcher 的 least_pending 策略使用，不发额外 HTTP 请求。
        """
        return ep.pending_count

    # ── 启动等待 ──────────────────────────────────────────────────────────────

    def wait_until_ready(self, timeout: float = 180.0, interval: float = 3.0) -> None:
        """阻塞直到所有 server 都通过 /health 探测。超时则抛出 RuntimeError。"""
        deadline = time.time() + timeout
        pending  = set(range(len(self.endpoints)))

        while pending:
            if time.time() > deadline:
                urls = [self.endpoints[i].url for i in pending]
                raise RuntimeError(
                    f"[ClusterTopology] Servers not ready within {timeout:.0f}s: {urls}"
                )
            for i in list(pending):
                ep   = self.endpoints[i]
                info = self._probe(ep)
                if info is not None:
                    ep.is_healthy     = True
                    ep.weight_version = info.get("weight_version", ep.weight_version)
                    ep.pending_count  = info.get("pending_count", 0)
                    pending.discard(i)
                    logger.info(f"[ClusterTopology] {ep.url} is ready.")
            if pending:
                urls = [self.endpoints[i].url for i in pending]
                logger.info(f"[ClusterTopology] Waiting for: {urls}")
                time.sleep(interval)

        logger.info(f"[ClusterTopology] All {len(self.endpoints)} servers ready.")

    # ── 后台健康检查 ──────────────────────────────────────────────────────────

    def start_health_watcher(self) -> None:
        """启动后台健康检查线程（daemon）。"""
        self._watcher = threading.Thread(
            target=self._health_loop,
            daemon=True,
            name="cluster-health-watcher",
        )
        self._watcher.start()
        logger.info(
            f"[ClusterTopology] Health watcher started "
            f"(interval={self.health_check_interval}s)."
        )

    def _health_loop(self) -> None:
        while not self._stop.wait(timeout=self.health_check_interval):
            futures = {
                self._probe_pool.submit(self._probe, ep): ep
                for ep in self.endpoints
            }
            for fut in as_completed(futures):
                ep   = futures[fut]
                info = fut.result()   # _probe 已吞掉网络异常，返回 None 表示失败
                if info is not None:
                    ep.is_healthy     = True
                    ep.weight_version = info.get("weight_version", ep.weight_version)
                    ep.pending_count  = info.get("pending_count", 0)
                    ep.last_error     = None
                else:
                    ep.is_healthy = False
                ep.last_health_ts = time.time()

    @staticmethod
    def _probe(ep: ServerEndpoint) -> Optional[dict]:
        """
        探测单个 endpoint：调用 /health，再尝试调 /pool_status 获取 pending 数。

        返回合并后的 info dict，失败返回 None。
        这是拓扑层唯一允许吞掉异常的地方——探测失败只标记不健康，不向上传播。
        """
        try:
            info = ep.health()
        except Exception as exc:
            ep.last_error = str(exc)
            logger.debug(f"[ClusterTopology] health probe failed {ep.url}: {exc}")
            return None

        # 顺带拉取 pending 数，失败不影响健康判定
        try:
            status = ep.pool_status()
            # pool_status 返回格式：{"pending": int, ...}
            info["pending_count"] = status.get("pending", 0)
        except Exception as exc:
            logger.debug(f"[ClusterTopology] pool_status probe failed {ep.url}: {exc}")
            info["pending_count"] = ep.pending_count   # 保留上次缓存值

        return info

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "num_servers":   len(self.endpoints),
            "healthy_count": sum(1 for ep in self.endpoints if ep.is_healthy),
            "servers": [
                {
                    "url":           ep.url,
                    "is_healthy":    ep.is_healthy,
                    "weight_version": ep.weight_version,
                    "pending_count": ep.pending_count,
                    "last_error":    ep.last_error,
                }
                for ep in self.endpoints
            ],
        }

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._stop.set()
        if self._watcher and self._watcher.is_alive():
            self._watcher.join(timeout=5)
        self._probe_pool.shutdown(wait=False)
        logger.info("[ClusterTopology] Shutdown complete.")