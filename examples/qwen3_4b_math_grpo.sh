#!/bin/bash
# ============================================================
# Qwen3-0.6B GRPO 训练脚本 —— 原版 SPMD 模式（双卡）
#
# GPU 分配：
#   GPU 0 + GPU 1 → FSDP 训练 + vLLM rollout（分时复用）
#
# 用法：bash examples/qwen3_0.6b_math_grpo.sh
# ============================================================

set -e

MODEL_PATH="${HOME}/huggingface/Qwen3-0.6B"
EXPERIMENT_NAME="qwen3_0.6b_math_grpo"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}"

CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.rollout_batch_size=64 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=32 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.ref.offload.offload_params=true \
    worker.rollout.n=3 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.6 \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=5 \
    trainer.val_freq=5 \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}"