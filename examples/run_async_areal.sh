#!/bin/bash
# ============================================================
# Qwen3-0.6B GRPO 训练脚本 —— 异步解耦模式（双卡 4090x2）
#
# GPU 分配：
#   GPU 0 → RolloutServer（vLLM 常驻, 90% memory, 异步生成）
#   GPU 1 → FSDP 训练（actor + ref, 从经验池消费）
#
# 核心改进：
#   - Rollout 和 Training 完全解耦
#   - Rollout 按自己节奏生成，不被 training 阻塞
#   - Training 从经验池（queue）中消费已完成的 rollout 结果
#   - 权重版本号用于过滤太旧的样本（staleness_tolerance=2）
#   - Pipeline 化：training 在更新时，rollout 已在生成下一批
#
# 用法：bash examples/run_async_mode.sh
# ============================================================

set -e

# 清理函数
cleanup() {
    echo -e "\n>>> 检测到中断信号，正在强制清理进程..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    pkill -9 -f "launch_server" || true
    pkill -9 -f "vllm" || true
    pkill -9 -f "verl.workers.rollout" || true
    echo ">>> 清理完毕，退出。"
    exit 1
}

trap cleanup SIGINT SIGTERM
export HF_ENDPOINT=https://hf-mirror.com

MODEL_PATH="${HOME}/huggingface/Qwen3-0.6B"
SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
EXPERIMENT_NAME="qwen3_0.6b_math_grpo_async"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}"

# ---------- Step 1: 启动异步 RolloutServer（GPU 0, 90% 显存）----------
echo ">>> 启动异步 RolloutServer on GPU 0 (90% memory) ..."
CUDA_VISIBLE_DEVICES=0 python -m verl.workers.rollout.launch_server \
    --model_path "${MODEL_PATH}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.6 \
    --prompt_length 1024 \
    --response_length 1024 \
    --dtype bf16 \
    --n 2 \
    --temperature 1.0 \
    --seed 1 \
    --pool_max_samples 4000 \
    --max_concurrent_prompts 32 \
    2>&1 | tee server.log &

SERVER_PID=$!
echo ">>> RolloutServer PID: ${SERVER_PID}"

# 等待 server 就绪
echo ">>> 等待 RolloutServer 启动（最多 120s）..."
for i in $(seq 1 60); do
    if curl -sf "http://${SERVER_HOST}:${SERVER_PORT}/health" > /dev/null 2>&1; then
        echo ">>> RolloutServer 已就绪！"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo ">>> 错误：RolloutServer 启动超时"
        kill ${SERVER_PID}
        exit 1
    fi
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Step 2: 启动训练（GPU 1）----------
echo ">>> 启动异步训练 on GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 python3 -m verl.trainer.main \
    config=examples/config.yaml\
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.val_batch_size=32 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.ref.offload.offload_params=false \
    worker.rollout.n=2 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.server_url="http://${SERVER_HOST}:${SERVER_PORT}" \
    algorithm.adv_estimator=grpo \
    algorithm.loss_type=areal \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=5 \
    trainer.val_freq=10 \
    trainer.staleness_tolerance=3 \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}"

TRAIN_EXIT=$?

# ---------- 清理 ----------
echo ">>> 训练结束（exit code: ${TRAIN_EXIT}），关闭 RolloutServer..."
kill ${SERVER_PID} 2>/dev/null
sleep 3
pkill -f "launch_server" 2>/dev/null
pkill -f "vllm" 2>/dev/null
pkill -f "verl.workers.rollout" 2>/dev/null
sleep 2
pkill -9 -f "launch_server" 2>/dev/null
pkill -9 -f "vllm" 2>/dev/null

REMAINING=$(ps aux | grep -E "launch_server|vllm" | grep -v grep | wc -l)
if [ ${REMAINING} -eq 0 ]; then
    echo ">>> 所有进程已清理完毕"
else
    echo ">>> 警告：仍有 ${REMAINING} 个相关进程存在，请手动检查"
    ps aux | grep -E "launch_server|vllm" | grep -v grep
fi

exit ${TRAIN_EXIT}
