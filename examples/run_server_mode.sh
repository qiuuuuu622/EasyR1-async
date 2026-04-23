#!/bin/bash
# ============================================================
# Qwen3-0.6B GRPO 训练脚本 —— Server 模式（双卡）
#
# GPU 分配：
#   GPU 0 → RolloutServer（vLLM 常驻）
#   GPU 1 → FSDP 训练（actor + ref）
#
# 用法：bash examples/qwen3_0.6b_math_grpo_server.sh
# ============================================================

set -e


# 定义清理函数
cleanup() {
    echo -e "\n>>> 检测到中断信号，正在强制清理进程..."
    # 杀掉通过 PID 记录的 Server
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    # 强力清理所有相关残留
    pkill -9 -f "launch_server" || true
    pkill -9 -f "vllm" || true
    pkill -9 -f "verl.workers.rollout" || true
    echo ">>> 清理完毕，退出。"
    exit 1
}

# 捕获 SIGINT (Ctrl+C) 和 SIGTERM 信号，并执行 cleanup
trap cleanup SIGINT SIGTERM

MODEL_PATH="${HOME}/huggingface/Qwen3-0.6B"
SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
EXPERIMENT_NAME="qwen3_0.6b_math_grpo_server"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}"

# ---------- Step 1: 启动 RolloutServer（GPU 0）----------
echo ">>> 启动 RolloutServer on GPU 0 ..."
CUDA_VISIBLE_DEVICES=0 python -m verl.workers.rollout.launch_server \
    --model_path "${MODEL_PATH}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.8 \
    --prompt_length 2048 \
    --response_length 4096 \
    --dtype bf16 \
    --n 5 \
    --temperature 1.0 \
    --seed 1 &

SERVER_PID=$!
echo ">>> RolloutServer PID: ${SERVER_PID}"

# 等待 server 就绪（vLLM 编译约需 60-90s）
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
echo ">>> 启动训练 on GPU 1 ..."
CUDA_VISIBLE_DEVICES=1 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false\
    worker.ref.offload.offload_params=false \
    worker.rollout.n=5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.server_url="http://${SERVER_HOST}:${SERVER_PORT}" \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=5 \
    trainer.val_freq=5 \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}"

TRAIN_EXIT=$?

# ---------- 清理 ----------
echo ">>> 训练结束（exit code: ${TRAIN_EXIT}），关闭 RolloutServer..."

# 杀主进程
kill ${SERVER_PID} 2>/dev/null

# 等待主进程退出
sleep 3

# 杀所有相关子进程
pkill -f "launch_server" 2>/dev/null
pkill -f "vllm" 2>/dev/null
pkill -f "verl.workers.rollout" 2>/dev/null

# 等待子进程退出
sleep 2

# 如果还有残留，强制kill
pkill -9 -f "launch_server" 2>/dev/null
pkill -9 -f "vllm" 2>/dev/null

# 确认清理结果
REMAINING=$(ps aux | grep -E "launch_server|vllm" | grep -v grep | wc -l)
if [ ${REMAINING} -eq 0 ]; then
    echo ">>> 所有进程已清理完毕"
else
    echo ">>> 警告：仍有 ${REMAINING} 个相关进程存在，请手动检查"
    ps aux | grep -E "launch_server|vllm" | grep -v grep
fi

exit ${TRAIN_EXIT}