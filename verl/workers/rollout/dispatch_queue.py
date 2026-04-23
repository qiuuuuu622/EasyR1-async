import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """队列已满，调用方应返回 HTTP 503 让客户端退避重试。"""


def _safe_len(obj: Any, default: int = 1) -> int:
    try:
        return len(obj)
    except Exception:
        return default


def _infer_prompt_len(gen_batch_item: Any) -> int:
    """
    尽量从 DataProto / batch["input_ids"] 中推断 prompt token 长度。
    约定 gen_batch_item 是 batch_size=1 的切片。
    """
    try:
        batch = getattr(gen_batch_item, "batch", None)
        if batch is None and isinstance(gen_batch_item, dict):
            batch = gen_batch_item.get("batch", None)
        if batch is None:
            return 1

        input_ids = batch["input_ids"]
        # 典型形状: [1, T]
        if hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
            return int(input_ids.shape[-1])

        # 回退
        return _safe_len(input_ids, default=1)
    except Exception:
        return 1


def _infer_expected_output_len(gen_batch_item: Any, default: int = 512) -> int:
    """
    尝试从 meta_info 里读 max_new_tokens / response_length，否则走默认值。
    """
    try:
        meta_info = getattr(gen_batch_item, "meta_info", None)
        if meta_info is None and isinstance(gen_batch_item, dict):
            meta_info = gen_batch_item.get("meta_info", None)
        if not isinstance(meta_info, dict):
            return default

        if "max_new_tokens" in meta_info:
            return int(meta_info["max_new_tokens"])
        if "response_length" in meta_info:
            return int(meta_info["response_length"])
        return default
    except Exception:
        return default


def _infer_n(gen_batch_item: Any, default: int = 1) -> int:
    """
    尝试从 meta_info 里读 n。
    """
    try:
        meta_info = getattr(gen_batch_item, "meta_info", None)
        if meta_info is None and isinstance(gen_batch_item, dict):
            meta_info = gen_batch_item.get("meta_info", None)
        if not isinstance(meta_info, dict):
            return default
        return int(meta_info.get("n", default))
    except Exception:
        return default


def _bucketize_prompt_len(prompt_len: int) -> int:
    """
    粗粒度长度分桶。你后续可以按压测再细调。
    """
    if prompt_len <= 256:
        return 0
    if prompt_len <= 512:
        return 1
    if prompt_len <= 1024:
        return 2
    if prompt_len <= 2048:
        return 3
    return 4


def _estimate_cost(
    prompt_len: int,
    expected_output_len: int,
    n: int,
    decode_weight: float,
) -> float:
    """
    一个简单可调的成本模型：
        cost = prompt_len + decode_weight * expected_output_len * n
    """
    return float(prompt_len) + float(decode_weight) * float(expected_output_len) * float(n)


@dataclass
class PendingPrompt:
    """单条 prompt 粒度的待推理任务。"""
    prompt_id: str
    request_id: str
    weight_version: int
    new_batch_item: Any   # DataProto 切片，batch_size=1，含训练字段
    gen_batch_item: Any   # DataProto 切片，batch_size=1，用于推理

    # ---- scheduling metadata ----
    prompt_len: int
    expected_output_len: int
    n: int
    est_cost: float
    bucket_id: int
    enqueue_ts: float
    priority: int = 0


class DispatchQueue:
    """
    线程安全的 pending prompt 分桶队列。

    提交侧（HTTP handler）:
      - submit(): 将请求拆成单条 PendingPrompt，按长度分桶入队

    消费侧（BatchScheduler）:
      - take_batch(): 按 max_prompts / max_cost 从最合适的桶中取出一个微批

    背压:
      - 队列容量按 prompt 数控制
      - 满了时 submit() 阻塞等待
    """

    def __init__(
        self,
        max_pending_prompts: int = 3000,
        decode_weight: float = 0.5,
        default_expected_output_len: int = 512,
        aging_weight: float = 0.01,
        max_cross_bucket_span: int = 1,
        default_n: int = 1
    ):
        self.default_n = default_n
        self.max_pending_prompts = max_pending_prompts
        self.decode_weight = decode_weight
        self.default_expected_output_len = default_expected_output_len
        self.aging_weight = aging_weight
        self.max_cross_bucket_span = max_cross_bucket_span

        self._buckets: dict[int, deque[PendingPrompt]] = {
            0: deque(),
            1: deque(),
            2: deque(),
            3: deque(),
            4: deque(),
        }
        self._bucket_ready_cost: dict[int, float] = {i: 0.0 for i in self._buckets}
        self._total_pending_prompts = 0
        self._total_pending_cost = 0.0

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._has_room = threading.Condition(self._lock)

        self._stats = {
            "submitted_requests": 0,
            "submitted_prompts": 0,
            "taken_batches": 0,
            "taken_prompts": 0,
        }

    # ------------------------------------------------------------------
    # 提交侧 API
    # ------------------------------------------------------------------

    def submit(
        self,
        new_batch,
        gen_batch,
        weight_version: int,
    ) -> str:
        """
        将一个请求拆解为逐条 PendingPrompt 并按长度分桶入队。

        背压：若队列已满，阻塞直到有空位。
        """
        request_id = str(uuid.uuid4())
        num_prompts = len(gen_batch)
        now = time.time()
        

        pending: list[PendingPrompt] = []
        for i in range(num_prompts):
            new_item = new_batch[i:i + 1]
            gen_item = gen_batch[i:i + 1]

            prompt_len = _infer_prompt_len(gen_item)
            expected_output_len = _infer_expected_output_len(
                gen_item, default=self.default_expected_output_len
            )
            n = _infer_n(gen_item, default=self.default_n)
            bucket_id = _bucketize_prompt_len(prompt_len)
            est_cost = _estimate_cost(
                prompt_len=prompt_len,
                expected_output_len=expected_output_len,
                n=n,
                decode_weight=self.decode_weight,
            )

            pending.append(
                PendingPrompt(
                    prompt_id=str(uuid.uuid4()),
                    request_id=request_id,
                    weight_version=weight_version,
                    new_batch_item=new_item,
                    gen_batch_item=gen_item,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len,
                    n=n,
                    est_cost=est_cost,
                    bucket_id=bucket_id,
                    enqueue_ts=now,
                )
            )

        with self._lock:
            if self._total_pending_prompts + num_prompts > self.max_pending_prompts:
                raise QueueFullError(
                    f"DispatchQueue full: "
                    f"{self._total_pending_prompts}/{self.max_pending_prompts} prompts"
                )

            for pp in pending:
                self._buckets[pp.bucket_id].append(pp)
                self._bucket_ready_cost[pp.bucket_id] += pp.est_cost
                self._total_pending_prompts += 1
                self._total_pending_cost += pp.est_cost

            self._stats["submitted_requests"] += 1
            self._stats["submitted_prompts"] += num_prompts
            self._not_empty.notify_all()

        logger.info(
            f"[DispatchQueue] submit request_id={request_id} "
            f"wv={weight_version} prompts={num_prompts} "
            f"pending_prompts={self._total_pending_prompts} "
            f"pending_cost={self._total_pending_cost:.1f}"
        )
        return request_id

    # ------------------------------------------------------------------
    # 消费侧 API
    # ------------------------------------------------------------------

    def take_batch(
        self,
        max_prompts: int,
        max_cost: float,
        preferred_weight_version: Optional[int] = None,
        max_cross_bucket_span: Optional[int] = None,
    ) -> list[PendingPrompt]:
        """
        从最合适的桶中取一个微批。

        规则：
          1. 优先选 backlog 大 + 等待久的桶
          2. 先尽量从主桶取
          3. 如有预算剩余，再从相邻桶补
          4. weight_version 仅做软偏好，不强过滤
        """
        if max_prompts <= 0 or max_cost <= 0:
            return []

        if max_cross_bucket_span is None:
            max_cross_bucket_span = self.max_cross_bucket_span

        with self._lock:
            if self._total_pending_prompts == 0:
                return []

            main_bucket = self._choose_main_bucket_locked(
                preferred_weight_version=preferred_weight_version
            )
            if main_bucket is None:
                return []

            batch: list[PendingPrompt] = []
            batch_cost = 0.0

            # 先取主桶
            self._drain_bucket_into_batch_locked(
                bucket_id=main_bucket,
                batch=batch,
                batch_cost_ref=[batch_cost],
                max_prompts=max_prompts,
                max_cost=max_cost,
                preferred_weight_version=preferred_weight_version,
            )
            batch_cost = sum(pp.est_cost for pp in batch)

            # 再从相邻桶补
            if len(batch) < max_prompts and batch_cost < max_cost:
                neighbors = self._neighbor_buckets(main_bucket, max_cross_bucket_span)
                for b in neighbors:
                    self._drain_bucket_into_batch_locked(
                        bucket_id=b,
                        batch=batch,
                        batch_cost_ref=[batch_cost],
                        max_prompts=max_prompts,
                        max_cost=max_cost,
                        preferred_weight_version=preferred_weight_version,
                    )
                    batch_cost = sum(pp.est_cost for pp in batch)
                    if len(batch) >= max_prompts or batch_cost >= max_cost:
                        break

            if batch:
                self._stats["taken_batches"] += 1
                self._stats["taken_prompts"] += len(batch)
                self._has_room.notify_all()

            return batch

    def _choose_main_bucket_locked(
        self,
        preferred_weight_version: Optional[int],
    ) -> Optional[int]:
        now = time.time()
        best_bucket = None
        best_score = float("-inf")

        for bucket_id, q in self._buckets.items():
            if not q:
                continue

            oldest_wait = max(0.0, now - q[0].enqueue_ts)
            score = self._bucket_ready_cost[bucket_id] + self.aging_weight * oldest_wait * max(1, len(q))

            # 软偏好：如果队头就匹配 preferred_weight_version，稍微加分
            if preferred_weight_version is not None and q[0].weight_version == preferred_weight_version:
                score *= 1.1

            if score > best_score:
                best_score = score
                best_bucket = bucket_id

        return best_bucket

    def _neighbor_buckets(self, center: int, span: int) -> list[int]:
        out = []
        for d in range(1, span + 1):
            left = center - d
            right = center + d
            if left in self._buckets:
                out.append(left)
            if right in self._buckets:
                out.append(right)
        return out

    def _drain_bucket_into_batch_locked(
        self,
        bucket_id: int,
        batch: list[PendingPrompt],
        batch_cost_ref: list[float],
        max_prompts: int,
        max_cost: float,
        preferred_weight_version: Optional[int],
    ) -> None:
        """
        从 bucket 队头尽量取，满足：
          - 不超过 max_prompts
          - 不超过 max_cost
        这里尽量保持 O(1) 队头消费，不做复杂重排。
        """
        q = self._buckets[bucket_id]
        if not q:
            return

        while q and len(batch) < max_prompts:
            pp = q[0]
            new_cost = batch_cost_ref[0] + pp.est_cost

            # 至少允许拿 1 个，避免大请求永久发不出去
            if batch and new_cost > max_cost:
                break

            # 软偏好：如果指定了 preferred_weight_version，且队头不匹配，
            # 不强行跳过，避免 O(n) 扫描带来的锁内开销。
            q.popleft()
            self._bucket_ready_cost[bucket_id] -= pp.est_cost
            self._total_pending_prompts -= 1
            self._total_pending_cost -= pp.est_cost

            batch.append(pp)
            batch_cost_ref[0] = new_cost

    # ------------------------------------------------------------------
    # 控制 API
    # ------------------------------------------------------------------

    def clear(self):
        with self._lock:
            for q in self._buckets.values():
                q.clear()
            for k in self._bucket_ready_cost:
                self._bucket_ready_cost[k] = 0.0
            self._total_pending_prompts = 0
            self._total_pending_cost = 0.0
            self._has_room.notify_all()
            self._not_empty.notify_all()

    def stats(self) -> dict:
        with self._lock:
            bucket_sizes = {f"bucket_{k}_prompts": len(v) for k, v in self._buckets.items()}
            bucket_costs = {f"bucket_{k}_cost": round(self._bucket_ready_cost[k], 2) for k in self._buckets}
            return {
                "pending_prompts": self._total_pending_prompts,
                "pending_cost": round(self._total_pending_cost, 2),
                "max_pending_prompts": self.max_pending_prompts,
                **bucket_sizes,
                **bucket_costs,
                **self._stats,
            }