"""
PPO Trainer with Ray-based single controller.

File layout
-----------
ppo_trainer/
├── ppo_trainer.py      ← this file
├── training_step.py    ← single-step forward/backward logic shared by both loops
├── prompt_producer.py  ← self-driven background prompt submission thread
└── config_validator.py ← pure config validation helpers
"""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ...single_controller.base import Worker
from ...single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ...single_controller.ray.base import create_colocated_worker_cls
from ...utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ...utils.logger import Tracker
from ...utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ...utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ...workers.fsdp_workers import FSDPWorker
from ...workers.reward import AutoRewardManager
from ...workers.rollout.cluster.cluster_dispatcher import ClusterDispatcher
from ...workers.rollout.cluster.cluster_topology import ClusterTopology
from .prompt_producer import PromptProducer
from ..config import PPOConfig
from .config_validator import validate_ppo_config
from ..core_algos import AdvantageEstimator, FixedKLController, get_kl_controller
from ..metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from .training_step import TrainingStep


# ─────────────────────────────────────────────────────────────────────────────
# Role + ResourcePoolManager (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class Role(IntEnum):
    Actor        = auto()
    Rollout      = auto()
    ActorRollout = auto()
    Critic       = auto()
    RefPolicy    = auto()
    RewardModel  = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self) -> None:
        for name, process_on_nodes in self.resource_pool_spec.items():
            pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=name,
            )
            self.resource_pool_dict[name] = pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        return sum(n for nodes in self.resource_pool_spec.values() for n in nodes)

    def _check_resource_available(self) -> None:
        available = ray.available_resources().get("GPU", 0)
        required  = self.get_num_gpus()
        print(f"[DEBUG] resource_pool_spec = {self.resource_pool_spec}")
        print(f"[DEBUG] Required GPUs: {required}, Available: {available}")
        if available < required:
            raise ValueError(
                f"Total available GPUs ({available}) < total required GPUs ({required})."
            )


# ─────────────────────────────────────────────────────────────────────────────
# RayPPOTrainer
# ─────────────────────────────────────────────────────────────────────────────

class RayPPOTrainer:
    """
    PPO trainer that runs on the driver process (single CPU/GPU node).

    Async vs Sync modes
    -------------------
    When ``config.worker.rollout.server_url(s)`` is set, the trainer runs in
    *async-decoupled* mode:
      - PromptProducer (background thread) continuously submits prompts to the
        rollout cluster via ClusterDispatcher. It is fully self-driven and only
        needs notify_weight_version() from Trainer after each weight push.
      - Trainer pulls completed samples, trains, and pushes weights on its own
        schedule. It does not manage submission timing or backpressure.
    Otherwise the classic synchronous rollout-then-train loop is used.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ) -> None:
        validate_ppo_config(config)

        self.config            = config
        self.tokenizer         = tokenizer
        self.processor         = processor
        self.train_dataloader  = train_dataloader
        self.val_dataloader    = val_dataloader
        self.reward_fn         = reward_fn
        self.val_reward_fn     = val_reward_fn
        self.role_worker_mapping      = role_worker_mapping
        self.resource_pool_manager    = resource_pool_manager
        self.ray_worker_group_cls     = ray_worker_group_cls

        self.global_step: int           = 0
        self.val_reward_score: float    = 0.0
        self.best_val_reward_score: float = -1.0
        self.best_global_step: Optional[int] = None

        self.hybrid_engine     = config.worker.hybrid_engine
        self.use_reward_model  = Role.RewardModel in role_worker_mapping
        self.use_critic        = config.algorithm.adv_estimator == AdvantageEstimator.GAE
        self.loss_type         = config.algorithm.loss_type

        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled. Set `kl_coef=0` to still log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        self.training_steps = self._compute_training_steps(config, train_dataloader)
        config.worker.actor.optim.training_steps  = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

        self._is_async_server_mode = bool(
            getattr(config.worker.rollout, "server_url", None)
            or getattr(config.worker.rollout, "server_urls", None)
        )
        self._staleness_tolerance = getattr(config.trainer, "staleness_tolerance", 2)
        if self._is_async_server_mode:
            print(f"[Async Mode] Enabled with staleness_tolerance={self._staleness_tolerance}")

        # PromptProducer instance — created in _async_fit, held here for
        # checkpoint access in _save_checkpoint / _load_checkpoint
        self._producer: Optional[PromptProducer] = None

    @staticmethod
    def _compute_training_steps(
        config: PPOConfig, train_dataloader: StatefulDataLoader
    ) -> int:
        if config.trainer.max_steps is not None:
            return config.trainer.max_steps
        if config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            return num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        return len(train_dataloader) * config.trainer.total_epochs

    # ─────────────────────────────────────────────────────────────────────────
    # Worker initialization
    # ─────────────────────────────────────────────────────────────────────────

    def init_workers(self) -> None:
        """Initialize resource pools and spawn worker groups."""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls: dict = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        if self.hybrid_engine:
            pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            self.resource_pool_to_cls[pool]["actor_rollout_ref"] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef],
                config=self.config.worker,
                role="actor_rollout_ref",
            )
        else:
            raise NotImplementedError("Only hybrid_engine is currently supported.")

        if self.use_critic:
            pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            self.resource_pool_to_cls[pool]["critic"] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.worker,
                role="critic",
            )

        if self.use_reward_model:
            pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            self.resource_pool_to_cls[pool]["rm"] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel],
                config=self.config.worker,
                role="reward",
            )

        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=pool, ray_cls_with_init=worker_dict_cls
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

        self._register_cleanup()

    def _register_cleanup(self) -> None:
        import atexit
        import signal
        import sys

        def cleanup():
            print("\n[verl] Termination detected. Cleaning up resources...")
            if self._producer is not None:
                try:
                    self._producer.stop()
                except Exception as exc:
                    print(f"[verl] Producer stop failed: {exc}")
            try:
                if hasattr(self, "actor_rollout_ref_wg"):
                    self.actor_rollout_ref_wg.shutdown_rollout.remote()
            except Exception as exc:
                print(f"[verl] Cleanup failed: {exc}")

        atexit.register(cleanup)

        def _signal_handler(sig, frame):
            cleanup()
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT,  _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except ValueError:
            pass  # Not in main thread

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self) -> None:
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step      = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder     = os.path.join(
            self.config.trainer.save_checkpoint_path,
            f"global_step_{self.global_step}",
        )
        actor_path = os.path.join(folder, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(
            actor_path, save_model_only=self.config.trainer.save_model_only
        )

        if self.use_critic:
            critic_path = os.path.join(folder, "critic")
            self.critic_wg.save_checkpoint(
                critic_path, save_model_only=self.config.trainer.save_model_only
            )

        # ── dataloader / producer state ──────────────────────────────────────
        # async mode: producer owns the dataloader, save via producer.state_dict()
        # sync mode:  trainer owns the dataloader, save directly
        dataloader_path = os.path.join(folder, "dataloader.pt")
        if self._producer is not None:
            torch.save(self._producer.state_dict(), dataloader_path)
        else:
            torch.save(self.train_dataloader.state_dict(), dataloader_path)

        tracker_info = {
            "best_global_step":     self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step":     self.global_step,
            "last_actor_path":      os.path.abspath(actor_path),
        }
        tracker_path = os.path.join(
            self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER
        )
        with open(tracker_path, "w") as f:
            json.dump(tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_path, tracker_info = find_latest_ckpt(
                self.config.trainer.save_checkpoint_path
            )
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step      = tracker_info.get("best_global_step", 0)
        else:
            load_path = None

        if load_path is None:
            return

        if "global_step_" not in load_path.strip(os.sep).split(os.sep)[-1]:
            raise ValueError("`load_checkpoint_path` must end with `global_step_*`.")

        print(f"Resuming from checkpoint: {load_path}")
        self.global_step = int(load_path.strip(os.sep).split("global_step_")[-1])
        self.actor_rollout_ref_wg.load_checkpoint(os.path.join(load_path, "actor"))

        if self.use_critic:
            self.critic_wg.load_checkpoint(os.path.join(load_path, "critic"))

        dataloader_path = os.path.join(load_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            state = torch.load(dataloader_path, weights_only=False)
            # async mode: producer not yet created here; stash state for later
            # sync mode: load directly into dataloader
            if self._is_async_server_mode:
                self._pending_producer_state = state
            else:
                self.train_dataloader.load_state_dict(
                    state if "dataloader" not in state else state["dataloader"]
                )
        else:
            print(f"No dataloader state at {dataloader_path}; starting from epoch 0.")

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self) -> dict[str, Any]:
        print("Starting validation...")
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst: dict[str, list] = defaultdict(list)
        length_metrics_lst: dict[str, list] = defaultdict(list)

        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = {
                **self.config.worker.rollout.val_override_config,
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps":  self.config.data.video_fps,
            }
            test_gen_batch, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_ref_wg.world_size
            )
            test_output = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output = unpad_dataproto(test_output, pad_size=pad_size * repeat_times)

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output)

            reward_tensor, reward_metrics = ray.get(
                self.val_reward_fn.compute_reward.remote(test_batch)
            )

            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in test_batch.batch["prompts"]
            ]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in test_batch.batch["responses"]
            ]
            scores = reward_tensor.sum(-1).cpu().tolist()

            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)
            reward_tensor_lst.append(reward_tensor)

            for k, v in reward_metrics.items():
                reward_metrics_lst[k].extend(v)
            for k, v in compute_length_metrics(test_batch).items():
                length_metrics_lst[k].append(v)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(
            sample_inputs, sample_outputs, sample_labels, sample_scores
        )
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        print("Validation complete.")
        return {
            "val/reward_score": self.val_reward_score,
            **{f"val/{k}_reward": v for k, v in reduce_metrics(reward_metrics_lst).items()},
            **{f"val_{k}": v for k, v in reduce_metrics(length_metrics_lst).items()},
        }

    def _maybe_log_val_generations(self, inputs, outputs, labels, scores) -> None:
        if self.config.trainer.val_generations_to_log <= 0:
            return
        samples = sorted(zip(inputs, outputs, labels, scores), key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        self.logger.log_generation(
            samples[: self.config.trainer.val_generations_to_log], self.global_step
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Shared batch utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _balance_batch(self, batch: DataProto, metrics: dict, prefix: str = "global_seqlen") -> None:
        """Reorder samples so each DP rank receives roughly equal token counts."""
        attn = batch.batch["attention_mask"]
        seqlens = attn.view(attn.shape[0], -1).sum(-1).tolist()
        world_size = self.actor_rollout_ref_wg.world_size
        partitions = get_seqlen_balanced_partitions(
            seqlens, k_partitions=world_size, equal_size=True
        )
        idx = torch.tensor([j for part in partitions for j in part])
        batch.reorder(idx)
        metrics.update(log_seqlen_unbalance(seqlens, partitions, prefix=prefix))

    def _maybe_validate_and_save(
        self,
        metrics: dict,
        timing_raw: dict,
        flush_cache_ref: Optional[list[bool]] = None,
    ) -> None:
        cfg  = self.config.trainer
        step = self.global_step

        if (
            self.val_reward_fn is not None
            and cfg.val_freq > 0
            and step % cfg.val_freq == 0
        ):
            with timer("validation", timing_raw):
                val_metrics = self._validate()
            metrics.update(val_metrics)
            if flush_cache_ref is not None:
                flush_cache_ref[0] = True

        if cfg.save_freq > 0 and step % cfg.save_freq == 0:
            with timer("save_checkpoint", timing_raw):
                self._save_checkpoint()

    def _log_step_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict) -> None:
        num_gpus = self.resource_pool_manager.get_num_gpus()
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        metrics.update(
            compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus)
        )
        self.logger.log(data=metrics, step=self.global_step)

    def _finalize_training(self, val_metrics: Optional[dict]) -> None:
        cfg = self.config.trainer
        if self.val_reward_fn is not None:
            needs_val = (
                val_metrics is None
                or cfg.val_freq <= 0
                or self.global_step % cfg.val_freq != 0
            )
            if needs_val:
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")
        if cfg.save_freq <= 0 or self.global_step % cfg.save_freq != 0:
            self._save_checkpoint()

    # ─────────────────────────────────────────────────────────────────────────
    # Sync rollout batch builder
    # ─────────────────────────────────────────────────────────────────────────

    def _make_batch_data(self, metrics: dict) -> DataProto:
        batch: Optional[DataProto] = None
        pending_metrics: dict[str, list] = defaultdict(list)
        num_attempts = 0

        print("Generating rollout batch...")
        while True:
            num_attempts += 1
            new_batch = self._rollout_one_mini_batch(pending_metrics)
            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch

            current_uniq = len(batch) // self.config.worker.rollout.n
            target_uniq  = self.config.data.rollout_batch_size

            if current_uniq < target_uniq:
                print(f"  {current_uniq} / {target_uniq} unique prompts collected.")
                max_try = self.config.trainer.max_try_make_batch
                if max_try > 0 and num_attempts >= max_try:
                    raise RuntimeError(
                        f"Gave up after {num_attempts} attempts. "
                        "Check online-filtering thresholds or data pipeline."
                    )
            else:
                print(f"  Rollout batch ready ({current_uniq} unique prompts).")
                if self.config.algorithm.online_filtering:
                    metrics.update(
                        {f"reward/{k}": v for k, v in reduce_metrics(pending_metrics).items()}
                    )
                total = target_uniq * self.config.worker.rollout.n
                return batch[:total]

    def _rollout_one_mini_batch(self, pending_metrics: dict) -> DataProto:
        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_dataloader)
            batch_dict = next(self.data_iterator)

        meta_info = {
            "min_pixels": self.config.data.min_pixels,
            "max_pixels": self.config.data.max_pixels,
            "video_fps":  self.config.data.video_fps,
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
        gen_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

        if self.config.algorithm.adv_estimator == "remax":
            baseline_gen = deepcopy(gen_batch)
            baseline_gen.meta_info["temperature"] = 0
            baseline_gen.meta_info["n"] = 1
            baseline_output = self.actor_rollout_ref_wg.generate_sequences(baseline_gen)
            new_batch = new_batch.union(baseline_output)
            reward_baseline, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            new_batch.pop(batch_keys=list(baseline_output.batch.keys()))
            new_batch.batch["reward_baselines"] = reward_baseline.sum(dim=-1)

        new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
        new_batch = new_batch.union(gen_output)

        if self.config.algorithm.online_filtering:
            new_batch = self._apply_online_filter(new_batch, pending_metrics)

        return new_batch

    def _apply_online_filter(self, batch: DataProto, pending_metrics: dict) -> DataProto:
        reward_tensor, reward_metrics = ray.get(
            self.reward_fn.compute_reward.remote(batch)
        )
        batch.batch["token_level_scores"] = reward_tensor
        for k, v in reward_metrics.items():
            pending_metrics[k].extend(v)

        cfg          = self.config.algorithm
        filter_scores = reward_metrics[cfg.filter_key]
        uids         = batch.non_tensor_batch["uid"]

        uid2scores: dict = defaultdict(list)
        for uid, score in zip(uids, filter_scores):
            uid2scores[uid].append(score)

        kept_uids = {
            uid for uid, scores in uid2scores.items()
            if cfg.filter_low < np.mean(scores) < cfg.filter_high
        }
        kept_idxs = [i for i, uid in enumerate(uids) if uid in kept_uids]
        if not kept_idxs:
            raise RuntimeError(
                "All samples filtered out. Check filter_low / filter_high thresholds."
            )
        return batch[kept_idxs]

    # ─────────────────────────────────────────────────────────────────────────
    # Training entry-point
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self) -> None:
        self.logger      = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[dict[str, Any]] = None
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        training_step = TrainingStep(self)

        if self._is_async_server_mode:
            self._async_fit(main_tqdm, val_metrics, training_step)
        else:
            self.data_iterator = iter(self.train_dataloader)
            self._sync_fit(main_tqdm, val_metrics, training_step)

    # ─────────────────────────────────────────────────────────────────────────
    # Synchronous training loop (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _sync_fit(self, main_tqdm, val_metrics, training_step: TrainingStep) -> None:
        while self.global_step < self.training_steps:
            self.global_step += 1
            metrics: dict[str, Any]   = {}
            timing_raw: dict[str, float] = {}

            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                self._balance_batch(batch, metrics)
                batch.meta_info["global_token_num"] = (
                    torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                )

                reward_ref = None
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                training_step.run(batch, metrics, timing_raw, reward_ref=reward_ref)
                self._maybe_validate_and_save(metrics, timing_raw)

            self._log_step_metrics(batch, metrics, timing_raw)
            main_tqdm.update()

        self._finalize_training(val_metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # Asynchronous training loop
    # ─────────────────────────────────────────────────────────────────────────

    def _async_fit(self, main_tqdm, val_metrics, training_step: TrainingStep) -> None:
        """
        Async-decoupled training loop.

        Trainer 只做三件事：
          1. pull_samples  — 从 rollout cluster 拉完整 batch
          2. training_step — 前向 + 反向 + 优化
          3. push_weights  — 推权重，通知 producer 版本号

        prompt 的提交完全由 PromptProducer 后台线程自驱，
        Trainer 不感知提交节奏、pending 数量、退避策略。
        """
        print("[Async Mode] Starting async-decoupled training loop...")

        rollout_cfg        = self.config.worker.rollout
        weight_sync_interval = getattr(self.config.trainer, "rollout_weight_sync_interval", 4)
        target_samples     = self.config.data.rollout_batch_size * rollout_cfg.n
        staleness_tolerance = self._staleness_tolerance

        # ── 构建 ClusterTopology + ClusterDispatcher ──────────────────────────
        server_urls = (
            [rollout_cfg.server_url]
            if getattr(rollout_cfg, "server_url", None)
            else list(rollout_cfg.server_urls)
        )
        topology = ClusterTopology(
            server_urls=server_urls,
            health_check_interval=rollout_cfg.health_check_interval,
        )
        topology.wait_until_ready()
        topology.start_health_watcher()

        dispatcher = ClusterDispatcher(
            topology=topology,
            strategy=getattr(rollout_cfg, "dispatch_strategy", "least_pending"),
        )

        # ── 构建 PromptProducer ───────────────────────────────────────────────
        producer = PromptProducer(
            dataloader=self.train_dataloader,
            dispatcher=dispatcher,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            backoff=getattr(self.config.trainer, "producer_backoff", 2.0),
        )
        self._producer = producer   # 供 _save_checkpoint 访问

        # 从 checkpoint 恢复 producer/dataloader 状态（_load_checkpoint 已暂存）
        pending_state = getattr(self, "_pending_producer_state", None)
        if pending_state is not None:
            producer.load_state_dict(pending_state)
            del self._pending_producer_state

        # ── 推初始权重 ────────────────────────────────────────────────────────
        print("[Async Mode] Pushing initial weights to rollout server...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        self.actor_rollout_ref_wg.push_weights_only(flush_cache=True)
        current_weight_version = 1
        producer.notify_weight_version(current_weight_version)

        # ── 启动后台提交线程 ──────────────────────────────────────────────────
        producer.start()
        print("[Async Mode] PromptProducer started, entering training loop...")

        # flush_cache: validation 后需要对 server 做一次完整缓存刷新
        flush_cache = [False]

        try:
            while self.global_step < self.training_steps:
                self.global_step += 1
                metrics:    dict[str, Any]   = {}
                timing_raw: dict[str, float] = {}

                with timer("step", timing_raw):

                    # ── 1. Pull a full batch from the experience pool ─────────
                    with timer("pull", timing_raw):
                        min_wv = max(0, current_weight_version - staleness_tolerance)
                        pull_results = self.actor_rollout_ref_wg.pull_samples(
                            target_samples=target_samples,
                            min_weight_version=min_wv,
                            timeout=1200.0,
                        )
                        pull_result  = pull_results[0]
                        batch: DataProto     = pull_result["data"]
                        pulled_wvs: list[int] = pull_result["weight_versions"]
                        pulled_count: int    = pull_result["total_samples"]

                    # async diagnostics
                    min_pulled_wv = min(pulled_wvs) if pulled_wvs else 0
                    metrics["async/min_experience_wv"]  = min_pulled_wv
                    metrics["async/max_staleness"]       = current_weight_version - min_pulled_wv
                    metrics["async/pulled_samples"]      = pulled_count
                    metrics["async/num_source_chunks"]   = len(pulled_wvs)
                    metrics["async/current_weight_version"] = current_weight_version

                    # truncate to exactly target_samples
                    if len(batch) > target_samples:
                        batch = batch[:target_samples]

                    # ── 2. Online filtering ───────────────────────────────────
                    if self.config.algorithm.online_filtering:
                        batch = self._apply_online_filter(batch, metrics)
                        if batch is None or len(batch) == 0:
                            print(f"[Step {self.global_step}] All samples filtered; skipping.")
                            continue

                    # ── 3. Balance batch across DP ranks ──────────────────────
                    self._balance_batch(batch, metrics)
                    batch.meta_info["global_token_num"] = (
                        torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    )

                    # ── 4. Async reward pre-launch ────────────────────────────
                    reward_ref = None
                    if "token_level_scores" not in batch.batch:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(batch)

                    # ── 5. Core update ────────────────────────────────────────
                    training_step.run(batch, metrics, timing_raw, reward_ref=reward_ref)

                    # ── 6. Push weights（时机和 flush 由 Trainer 决定）─────────
                    should_push = (
                        flush_cache[0]
                        or self.global_step % weight_sync_interval == 0
                        or self.global_step >= self.training_steps
                    )
                    if should_push:
                        with timer("push_weights", timing_raw):
                            self.actor_rollout_ref_wg.push_weights_only(
                                flush_cache=flush_cache[0]
                            )
                            current_weight_version += 1
                            # 通知 producer 新版本号，它下一次 submit 会携带
                            producer.notify_weight_version(current_weight_version)
                            print(
                                f"[Step {self.global_step}] Pushed weights "
                                f"(version={current_weight_version}, "
                                f"flush={flush_cache[0]})"
                            )
                        flush_cache[0] = False

                    # ── 7. Validation + checkpoint ────────────────────────────
                    self._maybe_validate_and_save(
                        metrics, timing_raw, flush_cache_ref=flush_cache
                    )

                self._log_step_metrics(batch, metrics, timing_raw)
                main_tqdm.update()

        finally:
            # 无论正常结束还是异常，都要停止后台线程
            producer.stop()
            self._producer = None

        self._finalize_training(val_metrics)
        print("[Async Mode] Training complete.")