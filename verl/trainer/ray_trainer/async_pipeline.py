
from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any
import numpy as np
from ...protocol import DataProto

if TYPE_CHECKING:
    from .ppo_trainer import RayPPOTrainer


class AsyncPromptPipeline:
    """
    Manages the async prompt pipeline for decoupled rollout.

    Parameters
    ----------
    trainer:
        Parent ``RayPPOTrainer`` instance.
    target_samples:
        Exact number of samples expected per training step
        (= ``rollout_batch_size * rollout.n``).
    staleness_tolerance:
        Maximum allowed weight-version lag for pulled samples.
    weight_sync_interval:
        How many steps between weight pushes to the rollout server.
    pipeline_depth_multiplier:
        Target in-flight samples = ``target_samples * this``.
    max_depth_multiplier:
        Hard cap on in-flight samples = ``target_samples * this``.
    low_watermark_multiplier:
        Trigger a refill when pending < ``target_samples * this``.
    """

    def __init__(
        self,
        trainer: "RayPPOTrainer",
        target_samples: int,
        staleness_tolerance: int = 2,
        weight_sync_interval: int = 4,
        pipeline_depth_multiplier: int = 4,
        max_depth_multiplier: int = 8,
        low_watermark_multiplier: int = 2,
    ) -> None:
        self._trainer = trainer
        self.target_samples = target_samples
        self.staleness_tolerance = staleness_tolerance
        self.weight_sync_interval = weight_sync_interval

        self._target_pending = target_samples * pipeline_depth_multiplier
        self._max_pending = target_samples * max_depth_multiplier
        self._low_watermark = target_samples * low_watermark_multiplier

        self._pending_samples: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fill(self) -> None:
        """
        Submit new prompts until the pipeline reaches the target depth
        (or the hard cap is hit).
        """
        while self._pending_samples < self._target_pending:
            added = self._submit_one_batch()
            self._pending_samples += added
            if self._pending_samples >= self._max_pending:
                break

    def maybe_fill(self) -> None:
        """
        Trigger a refill only when the pipeline falls below the low-watermark.
        Called at the end of each step to keep the server busy.
        """
        if self._pending_samples < self._low_watermark:
            self.fill()

    def pull(
        self,
        current_weight_version: int,
        metrics: dict[str, Any],
        timing_raw: dict[str, float],
    ) -> DataProto:
        """
        Block until ``target_samples`` are available, then return them as a
        single ``DataProto`` (truncated to exactly ``target_samples``).

        Also writes async diagnostics into *metrics* and *timing_raw*.
        """
        from ...utils.py_functional import timer

        min_wv = max(0, current_weight_version - self.staleness_tolerance)

        with timer("pull", timing_raw):
            pull_results = self._trainer.actor_rollout_ref_wg.pull_samples(
                target_samples=self.target_samples,
                min_weight_version=min_wv,
                timeout=1200.0,
            )
            pull_result = pull_results[0]
            batch: DataProto = pull_result["data"]
            pulled_wvs: list[int] = pull_result["weight_versions"]
            pulled_count: int = pull_result["total_samples"]

        # Accounting
        self._pending_samples = max(0, self._pending_samples - pulled_count)

        # Diagnostics
        min_pulled_wv = min(pulled_wvs) if pulled_wvs else 0
        metrics["async/min_experience_wv"] = min_pulled_wv
        metrics["async/max_staleness"] = current_weight_version - min_pulled_wv
        metrics["async/pulled_samples"] = pulled_count
        metrics["async/num_source_chunks"] = len(pulled_wvs)

        # Truncate to exactly target_samples
        if len(batch) > self.target_samples:
            batch = batch[: self.target_samples]

        return batch

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _submit_one_batch(self) -> int:
        """
        Pull one batch from the dataloader, build a gen_batch, submit it to
        the rollout server, and return the number of produced samples.
        """
        t = self._trainer
        cfg = t.config

        try:
            batch_dict = next(t.data_iterator)
        except StopIteration:
            t.data_iterator = iter(t.train_dataloader)
            batch_dict = next(t.data_iterator)

        meta_info = {
            "min_pixels": cfg.data.min_pixels,
            "max_pixels": cfg.data.max_pixels,
            "video_fps": cfg.data.video_fps,
        }
        new_batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
        )

        gen_batch = new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        gen_batch.meta_info["eos_token_id"] = t.tokenizer.eos_token_id
        gen_batch.meta_info["pad_token_id"] = t.tokenizer.pad_token_id

        num_prompts = len(new_batch)
        produced_samples = num_prompts * cfg.worker.rollout.n

        t.actor_rollout_ref_wg.submit_prompts(new_batch, gen_batch)
        return produced_samples