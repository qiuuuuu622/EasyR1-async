from ..config import PPOConfig
from ..core_algos import AdvantageEstimator


def validate_ppo_config(config: PPOConfig) -> None:
    """
    Validate PPO training configuration.
    Raises ValueError with a descriptive message on the first violation found.
    """
    _validate_batch_sizes(config)
    _validate_advantage_estimator(config)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_batch_sizes(config: PPOConfig) -> None:
    rollout_bs = config.data.rollout_batch_size
    actor_gbs = config.worker.actor.global_batch_size
    actor_mbs = config.worker.actor.micro_batch_size_per_device_for_experience
    rollout_n = config.worker.rollout.n

    if rollout_bs % actor_gbs != 0:
        raise ValueError(
            f"rollout_batch_size ({rollout_bs}) must be divisible by "
            f"actor.global_batch_size ({actor_gbs})."
        )

    if (rollout_bs * rollout_n) % actor_mbs != 0:
        raise ValueError(
            f"rollout_batch_size * rollout.n ({rollout_bs} * {rollout_n}) must be "
            f"divisible by actor.micro_batch_size_per_device_for_experience ({actor_mbs})."
        )

    if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        critic_gbs = config.worker.critic.global_batch_size
        critic_mbs = config.worker.critic.micro_batch_size_per_device_for_experience

        if rollout_bs % critic_gbs != 0:
            raise ValueError(
                f"rollout_batch_size ({rollout_bs}) must be divisible by "
                f"critic.global_batch_size ({critic_gbs})."
            )

        if (rollout_bs * rollout_n) % critic_mbs != 0:
            raise ValueError(
                f"rollout_batch_size * rollout.n ({rollout_bs} * {rollout_n}) must be "
                f"divisible by critic.micro_batch_size_per_device_for_experience ({critic_mbs})."
            )


def _validate_advantage_estimator(config: PPOConfig) -> None:
    estimator = config.algorithm.adv_estimator

    if estimator not in list(AdvantageEstimator):
        raise NotImplementedError(f"Unknown advantage estimator: {estimator}.")

    if estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO) and config.worker.rollout.n == 1:
        raise ValueError(
            f"GRPO and RLOO require config.worker.rollout.n > 1, "
            f"but got rollout.n = {config.worker.rollout.n}."
        )