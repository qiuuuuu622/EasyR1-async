import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReadyChunk:
    """一条 prompt 推理完成后，合并好的单个经验单元。"""
    data: Any
    weight_version: int
    ready_ts: float
    num_samples: int
    source_request_ids: list[str]
    group_id: str


class ExperiencePool:
    """
    线程安全的经验池。

    写入侧（BatchScheduler）调用 put() / put_many()；
    读取侧（HTTP /pull_samples handler）调用 pull()。

    容量管理：
      - 超过 max_ready_samples 时，从队头驱逐最旧 chunk
      - pull() 在返回前丢弃所有 weight_version < min_weight_version 的 chunk
    """

    def __init__(
        self,
        max_ready_samples: int = 5000,
    ):
        self.max_ready_samples = max_ready_samples

        self._chunks: deque[ReadyChunk] = deque()
        self._ready_samples = 0

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

        self._stats = {
            "completed_prompts": 0,
            "dropped_stale_chunks": 0,
            "evicted_chunks": 0,
        }

    # ------------------------------------------------------------------
    # 写入侧 API
    # ------------------------------------------------------------------

    def put(self, chunk: ReadyChunk) -> None:
        self.put_many([chunk])

    def put_many(self, chunks: list[ReadyChunk]) -> None:
        """
        批量写入 ready chunks。
        """
        if not chunks:
            return

        with self._lock:
            for chunk in chunks:
                self._chunks.append(chunk)
                self._ready_samples += chunk.num_samples
                self._stats["completed_prompts"] += 1

            # 超限驱逐，至少保留 1 个 chunk
            while self._ready_samples > self.max_ready_samples and len(self._chunks) > 1:
                old = self._chunks.popleft()
                self._ready_samples -= old.num_samples
                self._stats["evicted_chunks"] += 1
                logger.info(
                    f"[ExperiencePool] evict chunk group_id={old.group_id} "
                    f"wv={old.weight_version} samples={old.num_samples}"
                )

            self._not_empty.notify_all()

    # ------------------------------------------------------------------
    # 读取侧 API
    # ------------------------------------------------------------------

    def pull(
        self,
        target_samples: int,
        min_weight_version: int = 0,
        timeout: float = 1200.0,
        drain: bool = False,
    ) -> Optional[dict]:
        """
        从经验池取出样本。

        drain=False（默认）: 阻塞直到 ready_samples >= target_samples。
                             单 server 场景使用。
        drain=True:          立即返回当前池内所有可用样本，不等凑够。
                             多 server 场景使用，由 RolloutClusterClient 做全局聚合。
                             池内为空时返回 None，调用方稍后重试。
        """
        from ...protocol import DataProto

        deadline = time.time() + timeout

        with self._lock:
            self._drop_stale_locked(min_weight_version)

            if drain:
                # 有多少出多少，立即返回
                if not self._chunks:
                    return None
                collected, collected_count = self._collect_chunks_locked(target_samples)
            else:
                # 阻塞等到凑够
                while self._ready_samples < target_samples:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        logger.warning(
                            f"[ExperiencePool] pull timeout "
                            f"target={target_samples} ready={self._ready_samples}"
                        )
                        return None
                    self._not_empty.wait(timeout=min(remaining, 5.0))
                    self._drop_stale_locked(min_weight_version)
                collected, collected_count = self._collect_chunks_locked(target_samples)

        data_list = [c.data for c in collected]
        merged = data_list[0] if len(data_list) == 1 else DataProto.concat(data_list)
        return {
            "data": merged,
            "weight_versions": [c.weight_version for c in collected],
            "total_samples": collected_count,
        }

    def _collect_chunks_locked(self, max_samples: int) -> tuple:
        """锁内调用：从队头取出不超过 max_samples 条的 chunk 列表。"""
        collected = []
        collected_count = 0
        while collected_count < max_samples and self._chunks:
            chunk = self._chunks.popleft()
            self._ready_samples -= chunk.num_samples
            collected.append(chunk)
            collected_count += chunk.num_samples
        return collected, collected_count

    def _drop_stale_locked(self, min_weight_version: int) -> None:
        while self._chunks and self._chunks[0].weight_version < min_weight_version:
            old = self._chunks.popleft()
            self._ready_samples -= old.num_samples
            self._stats["dropped_stale_chunks"] += 1
            logger.info(
                f"[ExperiencePool] drop stale chunk group_id={old.group_id} "
                f"wv={old.weight_version} < min_wv={min_weight_version}"
            )

    # ------------------------------------------------------------------
    # 控制 API
    # ------------------------------------------------------------------

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._ready_samples = 0
            self._not_empty.notify_all()

    def stats(self) -> dict:
        with self._lock:
            return {
                "ready_chunks": len(self._chunks),
                "ready_samples": self._ready_samples,
                "max_ready_samples": self.max_ready_samples,
                **self._stats,
            }