import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .dispatch_queue import DispatchQueue, PendingPrompt
from .experience_pool import ExperiencePool, ReadyChunk
from .rollout_latency_tracker import NoOpLatencyTracker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

@dataclass
class BatchPlan:
    """
    描述一个待发射的微批及其输出切分方式。

    假设 batch 中第 i 个 PendingPrompt 的 n = p.n，
    那么 batched rollout 输出中，属于它的片段范围为 [start, end)。
    """
    prompts: list[PendingPrompt]
    output_ranges: list[tuple[int, int]]
    batch_cost: float
    batch_prompt_count: int


class BatchScheduler:
    """
    异步微批调度器，运行于 uvicorn 的 asyncio event loop 中。

    核心思路：
      - 不再单条 try_next() + create_task
      - 改为：take_batch() -> _run_batch()
      - 并发控制不再按 prompt 数，而按 inflight_cost

    can_schedule_fn:
      由外部注入，封装版本更新 / 暂停调度等业务门控。
    """

    def __init__(
        self,
        rollout,
        dispatch_queue: DispatchQueue,
        experience_pool: ExperiencePool,
        rollout_n: int,
        max_batch_prompts: int = 64,
        max_batch_cost: float = 32768.0,
        max_inflight_cost: float = 131072.0,
        batch_wait_timeout_s: float = 0.01,
        preferred_weight_version_fn: Optional[Callable[[], Optional[int]]] = None,
        latency_tracker=None,
    ):
        self.rollout = rollout
        self.dispatch_queue = dispatch_queue
        self.experience_pool = experience_pool
        self.rollout_n = rollout_n

        self.max_batch_prompts = max_batch_prompts
        self.max_batch_cost = max_batch_cost
        self.max_inflight_cost = max_inflight_cost
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.preferred_weight_version_fn = preferred_weight_version_fn
        self.latency_tracker = latency_tracker or NoOpLatencyTracker()

        self._stop = asyncio.Event()

        # asyncio 单线程内访问；加锁保护跨 task 更新的一致性
        self._inflight_cost: float = 0.0
        self._inflight_batches: int = 0
        self._state_lock = asyncio.Lock()
        self._batch_done_event = asyncio.Event()
        self._batch_done_event.set()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def run(self, can_schedule_fn: Callable[[], bool]) -> None:
        """
        主调度循环：
          - 根据 inflight_budget 决定还能不能发
          - 从 DispatchQueue 取一个微批
          - create_task(_run_batch(...))
        """
        while not self._stop.is_set():
            try:
                if not can_schedule_fn():
                    await asyncio.sleep(0.1)
                    continue

                budget_left = self.max_inflight_cost - self.inflight_cost
                if budget_left <= 0:
                    self._batch_done_event.clear()
                    try:
                        await asyncio.wait_for(self._batch_done_event.wait(), timeout=0.05)
                    except asyncio.TimeoutError:
                        pass
                    continue

                preferred_weight_version = (
                    self.preferred_weight_version_fn()
                    if self.preferred_weight_version_fn is not None
                    else None
                )

                batch = self.dispatch_queue.take_batch(
                    max_prompts=self.max_batch_prompts,
                    max_cost=min(self.max_batch_cost, budget_left),
                    preferred_weight_version=preferred_weight_version,
                )

                if not batch:
                    await asyncio.sleep(self.batch_wait_timeout_s)
                    continue

                batch_plan = self._build_batch_plan(batch)

                async with self._state_lock:
                    self._inflight_cost += batch_plan.batch_cost
                    self._inflight_batches += 1
                    self._batch_done_event.clear()

                asyncio.create_task(self._run_batch(batch_plan))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[BatchScheduler] loop error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # 批计划
    # ------------------------------------------------------------------

    def _build_batch_plan(self, batch: list[PendingPrompt]) -> BatchPlan:
        output_ranges: list[tuple[int, int]] = []
        cursor = 0
        total_cost = 0.0

        for pp in batch:
            n = self.rollout_n
            start = cursor
            end = cursor + n
            output_ranges.append((start, end))
            cursor = end
            total_cost += pp.est_cost

        return BatchPlan(
            prompts=batch,
            output_ranges=output_ranges,
            batch_cost=total_cost,
            batch_prompt_count=len(batch),
        )

    # ------------------------------------------------------------------
    # 单个微批执行
    # ------------------------------------------------------------------

    async def _run_batch(self, batch_plan: BatchPlan) -> None:
        tracker = self.latency_tracker
        start_ts = time.time()

        try:
            for pp in batch_plan.prompts:
                tracker.on_submit(pp.prompt_id, pp.request_id, pp.weight_version)

            # 1) 拼 batch
            batched_gen_batch = self._concat_dataprotos(
                [pp.gen_batch_item for pp in batch_plan.prompts]
            )

            # 2) 一次 generate
            for pp in batch_plan.prompts:
                tracker.on_slot_enter(pp.prompt_id)
                tracker.on_gen_start(pp.prompt_id)

            output = await self.rollout.generate_sequences(prompts=batched_gen_batch)

            for pp in batch_plan.prompts:
                tracker.on_gen_end(pp.prompt_id)

            # 3) 拆 batch -> chunks
            chunks: list[ReadyChunk] = []
            for pp, (start, end) in zip(batch_plan.prompts, batch_plan.output_ranges):
                output_slice = output[start:end]
                chunk = self._build_chunk(pp, output_slice)
                chunks.append(chunk)

            # 4) 批量写池
            self.experience_pool.put_many(chunks)

            for pp in batch_plan.prompts:
                tracker.on_complete(pp.prompt_id)

        except Exception as e:
            prompt_ids = [pp.prompt_id for pp in batch_plan.prompts]
            logger.error(
                f"[BatchScheduler] batch FAILED prompt_ids={prompt_ids}: {e}",
                exc_info=True,
            )
        finally:
            async with self._state_lock:
                self._inflight_cost -= batch_plan.batch_cost
                if self._inflight_cost < 0:
                    self._inflight_cost = 0.0

                self._inflight_batches -= 1
                if self._inflight_batches < 0:
                    self._inflight_batches = 0

                self._batch_done_event.set()

    # ------------------------------------------------------------------
    # 批内数据拼接 / 经验构造
    # ------------------------------------------------------------------

    def _concat_dataprotos(self, data_list: list):
        """
        统一在这里拼 batch。默认 DataProto 提供 concat。
        """
        if not data_list:
            raise ValueError("Empty data_list in _concat_dataprotos().")
        if len(data_list) == 1:
            return data_list[0]

        from ...protocol import DataProto
        return DataProto.concat(data_list)

    def _build_chunk(self, pp: PendingPrompt, output_slice) -> ReadyChunk:
        
        """
        将属于该 PendingPrompt 的输出切片，构造成单个 ReadyChunk。
        """
        merged = pp.new_batch_item.repeat(
            repeat_times=self.rollout_n,
            interleave=True,
        )
        merged = merged.union(output_slice)

        return ReadyChunk(
            data=merged,
            weight_version=pp.weight_version,
            ready_ts=time.time(),
            num_samples=len(merged),
            source_request_ids=[pp.request_id],
            group_id=pp.prompt_id,
        )

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def inflight_cost(self) -> float:
        return self._inflight_cost

    def stats(self) -> dict:
        return {
            "inflight_batches": self._inflight_batches,
            "inflight_cost": round(self._inflight_cost, 2),
            "max_batch_prompts": self.max_batch_prompts,
            "max_batch_cost": self.max_batch_cost,
            "max_inflight_cost": self.max_inflight_cost,
        }