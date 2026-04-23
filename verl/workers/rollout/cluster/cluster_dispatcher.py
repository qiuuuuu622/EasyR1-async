"""
ClusterDispatcher — 分发决策层。

职责（仅此而已）:
  - 根据策略（round_robin / least_pending）从健康 server 中选目标
  - 调用 ServerEndpoint.submit_prompts()
  - 把 HTTP 层的多种失败情况归一化为 Producer 能理解的两种语义：
      ServerBusyError  → 所有健康 server 当前都满，调用方退避后重试
      RuntimeError     → 无任何健康 server，严重错误

  内部处理（对 Producer 透明）:
      某 server 503     → ServerBusyError，记录，继续尝试其他 server
      某 server 网络失败 → NetworkError，标记 is_healthy=False，换 server 重试
      所有 server 都 503 → 向上抛 ServerBusyError

不做:
  - 健康检查（ClusterTopology 后台线程负责）
  - FSDP / NCCL（RolloutClusterClient 负责）
  - 序列化（在调用处完成）
  - 数据聚合拉取（SampleAggregator / RolloutClusterClient.pull_samples 负责）
"""

from __future__ import annotations

import logging
import pickle
import threading
from typing import Literal

from .cluster_topology import ClusterTopology
from .server_endpoint import ServerBusyError, NetworkError, ServerEndpoint

logger = logging.getLogger(__name__)

Strategy = Literal["round_robin", "least_pending"]


class ClusterDispatcher:
    """
    Producer 的唯一出口。

    线程安全：round_robin 的计数器用 Lock 保护；least_pending 读的
    pending_count 由 ClusterTopology 后台线程写入，无锁读（Python GIL
    保证整数赋值原子性，轻微的读写竞争不影响路由质量）。
    """

    def __init__(
        self,
        topology: ClusterTopology,
        strategy: Strategy = "least_pending",
    ):
        if strategy not in ("round_robin", "least_pending"):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        self._topology = topology
        self._strategy = strategy
        self._rr_index = 0
        self._rr_lock  = threading.Lock()

    # ── 唯一对外接口 ──────────────────────────────────────────────────────────

    def submit(
        self,
        new_batch,          # DataProto
        gen_batch,          # DataProto
        weight_version: int,
    ) -> None:
        """
        选一个健康 server 并提交 (new_batch, gen_batch, weight_version)。

        失败语义（Producer 只需处理这两种）：
            ServerBusyError  → 所有健康 server 当前都满，调用方退避后重试
            RuntimeError     → 无任何健康 server，严重错误，通常不可恢复

        内部重试逻辑（对 Producer 透明）：
            单个 server 503  → 换下一个健康 server 重试
            单个 server 网络失败 → 标记 unhealthy，换 server 重试
        """
        healthy = self._topology.healthy_endpoints()
        if not healthy:
            raise RuntimeError(
                "[ClusterDispatcher] No healthy servers available for submit."
            )

        gen_batch.meta_info["weight_version"] = weight_version
        payload_bytes = pickle.dumps((new_batch, gen_batch))
        busy_count    = 0

        # 按策略排序，让优先目标排在前面，依次尝试
        ordered = self._order_endpoints(healthy)

        for ep in ordered:
            try:
                ep.submit_prompts(payload_bytes)
                logger.debug(
                    f"[ClusterDispatcher] submitted → {ep.url} "
                    f"wv={weight_version} "
                    f"({len(payload_bytes) / 1024:.1f} KB)"
                )
                return   # 成功，立即返回
            except ServerBusyError:
                busy_count += 1
                logger.debug(f"[ClusterDispatcher] {ep.url} busy, trying next server.")
            except NetworkError as e:
                ep.is_healthy = False
                logger.warning(
                    f"[ClusterDispatcher] {ep.url} network error, "
                    f"marked unhealthy: {e}"
                )

        # 走到这里：所有 server 要么 busy，要么 unhealthy
        if busy_count > 0:
            # 至少有 server 还活着但满了，背压信号
            raise ServerBusyError(
                f"[ClusterDispatcher] All {len(ordered)} healthy server(s) are busy."
            )
        else:
            # 所有 server 都因网络错误被标记 unhealthy
            raise RuntimeError(
                "[ClusterDispatcher] All servers became unhealthy during submit."
            )

    # ── 内部路由策略 ──────────────────────────────────────────────────────────

    def _order_endpoints(self, healthy: list[ServerEndpoint]) -> list[ServerEndpoint]:
        """
        按策略对健康 endpoint 排序，返回有序列表供逐一尝试。

        round_robin:   全局轮转，每次 submit 推进一格
        least_pending: 按 ClusterTopology 缓存的 pending_count 升序
                       不发额外 HTTP，使用探测时缓存的值
        """
        if self._strategy == "round_robin":
            with self._rr_lock:
                # 以当前 rr_index 为起点做一次旋转，保证每次从不同位置开始
                start = self._rr_index % len(healthy)
                self._rr_index += 1
            return healthy[start:] + healthy[:start]

        # least_pending
        return sorted(healthy, key=lambda ep: self._topology.pending_of(ep))