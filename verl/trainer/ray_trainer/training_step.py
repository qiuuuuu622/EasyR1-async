# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TrainingStep: encapsulates the "forward + backward" logic of one PPO step.

Both _sync_fit and _async_fit call TrainingStep.run(), which covers:
  1. compute old log-probs
  2. compute ref log-probs  (optional)
  3. compute critic values  (optional)
  4. compute rewards + KL penalty + advantage
  5. update critic          (optional)
  6. update actor

This class holds *no mutable state* beyond weak references to the worker
groups and config — callers own the batch and metrics dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import ray

from ...protocol import DataProto
from ...utils.py_functional import timer
from ..core_algos import compute_advantage_return, AdvantageEstimator
from ..metrics import reduce_metrics
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Avoid circular imports; only used for type hints.
    from .ppo_trainer import RayPPOTrainer


class TrainingStep:
    """
    Stateless helper that executes one PPO update step.

    Parameters
    ----------
    trainer:
        The parent ``RayPPOTrainer`` instance.  We hold a reference so we can
        reach worker groups, config, and KL controller without threading long
        argument lists through every method.
    """

    def __init__(self, trainer: "RayPPOTrainer") -> None:
        self._trainer = trainer

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        timing_raw: dict[str, float],
        reward_ref=None,
    ) -> None:
        """
        Execute one full PPO update in-place.

        Parameters
        ----------
        batch:
            The experience batch for this step.  Will be mutated (new keys
            are added via ``union``).
        metrics:
            Accumulated metrics dict; new entries are written in-place.
        timing_raw:
            Accumulated timing dict; new entries are written in-place.
        reward_ref:
            Optional pre-launched Ray future for ``reward_fn.compute_reward``.
            Pass it when the reward computation was started before this call so
            we can overlap it with log-prob computation.
        """
        self._compute_log_probs(batch, metrics, timing_raw)
        self._compute_ref_log_probs(batch, metrics, timing_raw)
        self._compute_values(batch, metrics, timing_raw)
        self._compute_advantage(batch, metrics, timing_raw, reward_ref)
        self._update_critic(batch, metrics, timing_raw)
        self._update_actor(batch, metrics, timing_raw)

    # ─────────────────────────────────────────────────────────────────────────
    # Private step helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_log_probs(self, batch, metrics, timing_raw):
        t = self._trainer
        with timer("old", timing_raw):
            prox_log_probs = t.actor_rollout_ref_wg.compute_log_probs(batch)
            batch.union(prox_log_probs)

    def _compute_ref_log_probs(
        self, batch: DataProto, metrics: dict, timing_raw: dict
    ) -> None:
        t = self._trainer
        with timer("ref", timing_raw):
            ref_log_probs = t.actor_rollout_ref_wg.compute_ref_log_probs(batch)
            batch.union(ref_log_probs)

    def _compute_values(
        self, batch: DataProto, metrics: dict, timing_raw: dict
    ) -> None:
        t = self._trainer
        if not t.use_critic:
            return
        with timer("values", timing_raw):
            values = t.critic_wg.compute_values(batch)
            batch.union(values)

    def _compute_advantage(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
        reward_ref=None,
    ) -> None:
        """Resolve rewards, apply KL penalty, and compute advantages."""
        t = self._trainer
        cfg = t.config.algorithm

        with timer("adv", timing_raw):
            # ── Resolve rewards ──────────────────────────────────────────────
            if "token_level_scores" not in batch.batch:
                reward_tensor, reward_metrics = ray.get(reward_ref)
                batch.batch["token_level_scores"] = reward_tensor
                metrics.update(
                    {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                )

            # ── KL penalty (or pass-through) ─────────────────────────────────
            if not cfg.use_kl_loss and t.use_reference_policy:
                batch, kl_metrics = _apply_kl_penalty(batch, t.kl_ctrl, cfg.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # ── Advantage ────────────────────────────────────────────────────
            _compute_advantage(batch, cfg.adv_estimator, cfg.gamma, cfg.lam)

    def _update_critic(
        self, batch: DataProto, metrics: dict, timing_raw: dict
    ) -> None:
        t = self._trainer
        if not t.use_critic:
            return
        with timer("update_critic", timing_raw):
            critic_output = t.critic_wg.update_critic(batch)
        metrics.update(reduce_metrics(critic_output.non_tensor_batch))

    def _update_actor(
        self, batch: DataProto, metrics: dict, timing_raw: dict
    ) -> None:
        t = self._trainer
        if t.config.trainer.critic_warmup > t.global_step:
            return
        with timer("update_actor", timing_raw):
            actor_output = t.actor_rollout_ref_wg.update_actor(batch)
        metrics.update(reduce_metrics(actor_output.non_tensor_batch))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level pure functions (re-exported for use in trainer.py as well)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_kl_penalty(batch: DataProto, kl_ctrl, kl_penalty: str):
    """Thin wrapper kept here so both TrainingStep and trainer.py share one impl."""
    from ...utils import torch_functional as VF
    from ..core_algos import compute_kl
    import torch

    token_level_scores = batch.batch["token_level_scores"]
    batch_size = batch.batch.batch_size[0]
    response_mask = batch.batch["response_mask"]

    kld = compute_kl(batch.batch["old_log_probs"], batch.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    batch.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return batch, metrics


def _compute_advantage(batch: DataProto, adv_estimator: AdvantageEstimator, gamma: float, lam: float) -> None:
    """Compute advantages and returns in-place on ``batch``."""
    from ..core_algos import compute_advantage_return

    adv_inputs = {
        "token_level_rewards": batch.batch["token_level_rewards"],
        "response_mask": batch.batch["response_mask"],
        "index": batch.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in batch.batch:
        adv_inputs["values"] = batch.batch["values"]
    if "reward_baselines" in batch.batch:
        adv_inputs["reward_baselines"] = batch.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = returns