"""
ServerEndpoint — 单个 RolloutServer 的 HTTP 传输层。

职责（仅此而已）:
  - 持有 url、健康状态、weight_version 等元数据
  - 提供对 /health /update_weights /submit_prompts /pull_samples /generate
    的类型化 HTTP 调用
  - 把原始 HTTP 状态码翻译为业务异常，供上层做路由决策

异常语义（唯一的语义翻译点）:
  - 503  → ServerBusyError   DispatchQueue 满，server 正常但暂时无法接收
                              不标记 is_healthy=False，调用方退避后可重试同一 server
  - 5xx  → NetworkError      server 侧错误，调用方应标记 is_healthy=False
  - 连接失败/超时 → NetworkError  同上
  - 其余 4xx → requests.HTTPError  调用方按需处理

调用者负责决定往哪个 endpoint 发、失败了怎么办。
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from ....protocol import DataProto

logger = logging.getLogger(__name__)

# ── 超时常量 ──────────────────────────────────────────────────────────────────
WEIGHT_PUSH_TIMEOUT = 600   # s
GENERATE_TIMEOUT    = 600
SUBMIT_TIMEOUT      = 60
PULL_TIMEOUT        = 1800
HEALTH_TIMEOUT      = 5

_OCTET = {"Content-Type": "application/octet-stream"}


# ── 业务异常（语义翻译结果）────────────────────────────────────────────────────

class ServerBusyError(Exception):
    """
    DispatchQueue 满，server 返回 503。
    server 本身是健康的，只是当前无法接收新 prompt。
    调用方应退避后重试，不应标记该 server 为 unhealthy。
    """

class NetworkError(Exception):
    """
    server 侧错误（5xx）或网络连接失败。
    调用方应将该 server 标记为 unhealthy，并路由到其他 server。
    """


# ── ServerEndpoint ────────────────────────────────────────────────────────────

@dataclass
class ServerEndpoint:
    """
    一个 RolloutServer 实例的连接句柄 + 健康元数据。

    只负责「如何发」，不管「发给谁」「失败了怎办」。
    健康状态由 ClusterTopology 的后台线程写入，Dispatcher 只读。
    """

    url: str
    index: int
    weight_version: int       = -1
    is_healthy: bool          = True
    last_error: Optional[str] = None
    last_health_ts: float     = field(default_factory=time.time)
    # ClusterTopology 探测时顺带缓存，供 least_pending 策略使用
    pending_count: int        = 0

    def __post_init__(self):
        self.url = self.url.rstrip("/")

    def __repr__(self) -> str:
        status = "UP" if self.is_healthy else "DOWN"
        return (
            f"Server[{self.index}]({self.url}) {status} "
            f"wv={self.weight_version} pending={self.pending_count}"
        )

    # ── 公开 HTTP 方法 ────────────────────────────────────────────────────────

    def health(self) -> dict:
        """GET /health → dict。连接失败直接抛出（由 ClusterTopology._probe 吞掉）。"""
        resp = requests.get(f"{self.url}/health", timeout=HEALTH_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def pool_status(self) -> dict:
        """GET /pool_status → dict。"""
        resp = requests.get(f"{self.url}/pool_status", timeout=HEALTH_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def push_weights(self, payload_bytes: bytes) -> dict:
        """
        POST /update_weights。

        payload_bytes 已由调用方序列化，此处只负责传输。
        返回 server 的 JSON 响应。
        网络失败抛 NetworkError。
        """
        try:
            resp = requests.post(
                f"{self.url}/update_weights",
                data=payload_bytes,
                headers=_OCTET,
                timeout=WEIGHT_PUSH_TIMEOUT,
            )
            if resp.status_code >= 500:
                raise NetworkError(
                    f"{self.url} push_weights failed with {resp.status_code}: {resp.text[:200]}"
                )
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"{self.url} push_weights connection error: {e}") from e

    def submit_prompts(self, payload_bytes: bytes) -> dict:
        """
        POST /submit_prompts。

        异常语义：
            503 → ServerBusyError  （不标 unhealthy，调用方退避后可重试）
            5xx → NetworkError     （调用方标记 unhealthy）
            连接失败 → NetworkError
        """
        try:
            resp = requests.post(
                f"{self.url}/submit_prompts",
                data=payload_bytes,
                headers=_OCTET,
                timeout=SUBMIT_TIMEOUT,
            )
            if resp.status_code == 503:
                raise ServerBusyError(
                    f"{self.url} DispatchQueue full (503)"
                )
            if resp.status_code >= 500:
                raise NetworkError(
                    f"{self.url} submit_prompts error {resp.status_code}: {resp.text[:200]}"
                )
            resp.raise_for_status()
            return resp.json()
        except ServerBusyError:
            raise   # 原样向上，不包装
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"{self.url} submit_prompts connection error: {e}") from e

    def pull_samples(self, params: dict, server_timeout: float) -> dict:
        """
        POST /pull_samples（阻塞直到 server 凑够样本）。

        params: {"target_samples": int, "min_weight_version": int, "timeout": float}
        返回反序列化后的 dict（含 "data": DataProto）。
        """
        try:
            resp = requests.post(
                f"{self.url}/pull_samples",
                data=pickle.dumps(params),
                headers=_OCTET,
                timeout=server_timeout + 30,   # 略大于 server 端 timeout
            )
            if resp.status_code == 408:
                raise TimeoutError(f"{self.url} pull timed out (408)")
            if resp.status_code >= 500:
                raise NetworkError(
                    f"{self.url} pull_samples error {resp.status_code}: {resp.text[:200]}"
                )
            resp.raise_for_status()
            return pickle.loads(resp.content)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"{self.url} pull_samples connection error: {e}") from e

    def generate(self, payload_bytes: bytes) -> "DataProto":
        """POST /generate（同步推理，用于 validation）。返回 DataProto。"""
        try:
            resp = requests.post(
                f"{self.url}/generate",
                data=payload_bytes,
                headers=_OCTET,
                timeout=GENERATE_TIMEOUT,
            )
            if resp.status_code >= 500:
                raise NetworkError(
                    f"{self.url} generate error {resp.status_code}: {resp.text[:200]}"
                )
            resp.raise_for_status()
            return pickle.loads(resp.content)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"{self.url} generate connection error: {e}") from e