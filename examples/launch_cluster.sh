#!/bin/bash
# ============================================================
# Qwen3-0.6B GRPO 训练脚本 —— 多卡异步解耦模式
#
# 修改以下两个变量即可切换 GPU 分配：
#   ROLLOUT_GPUS: 给 rollout 的 GPU 列表，每个元素一个 SGLang 实例
#   TRAIN_GPUS:   给训练的 GPU，逗号分隔
#
# 示例：
#   1卡rollout + 1卡训练（当前）:        ROLLOUT_GPUS=(0)            TRAIN_GPUS="1"
#   4卡rollout + 2卡训练:                ROLLOUT_GPUS=(0 1 2 3)      TRAIN_GPUS="4,5"
#   6卡rollout + 2卡训练:                ROLLOUT_GPUS=(0 1 2 3 4 5)  TRAIN_GPUS="6,7"
#
# 用法：bash examples/run_async_cluster.sh
# ============================================================

set -e

# ── 【修改这里】GPU 分配 ──────────────────────────────────────
ROLLOUT_GPUS=(0)       # rollout GPU 列表，空格分隔；每个 GPU 启动一个独立 server
TRAIN_GPUS="1"         # 训练 GPU，多卡用逗号: "2,3"
# ─────────────────────────────────────────────────────────────

NUM_ROLLOUT_SERVERS=${#ROLLOUT_GPUS[@]}

# ── 基础配置 ─────────────────────────────────────────────────
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="${HOME}/huggingface/Qwen3-0.6B"
SERVER_HOST="127.0.0.1"
BASE_PORT=8000

EXPERIMENT_NAME="qwen3_0_6b_math_grpo_cluster"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}"

# ── 清理函数 ─────────────────────────────────────────────────
SERVER_PIDS=()

cleanup() {
    echo -e "\n>>> 检测到中断信号，正在清理..."
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -9 -f "launch_server" 2>/dev/null || true
    pkill -9 -f "sglang"        2>/dev/null || true
    echo ">>> 清理完毕，退出。"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ── Step 1: 启动 RolloutServer ────────────────────────────────
echo ">>> 启动 ${NUM_ROLLOUT_SERVERS} 个 RolloutServer..."
echo ">>>   ROLLOUT_GPUS=(${ROLLOUT_GPUS[*]})"
echo ">>>   TRAIN_GPUS=${TRAIN_GPUS}"

SERVER_URLS=()

for i in "${!ROLLOUT_GPUS[@]}"; do
    GPU=${ROLLOUT_GPUS[$i]}
    PORT=$((BASE_PORT + i))
    LOG="server_gpu${GPU}.log"

    CUDA_VISIBLE_DEVICES=$GPU python -m verl.workers.rollout.launch_server \
        --model_path              "${MODEL_PATH}" \
        --host                    "${SERVER_HOST}" \
        --port                    "${PORT}" \
        --tensor_parallel_size    1 \
        --gpu_memory_utilization  0.7 \
        --prompt_length           2048 \
        --response_length         2048 \
        --dtype                   bf16 \
        --n                       5 \
        --temperature             1.0 \
        --seed                    $((1 + i)) \
        --pool_max_samples        4000 \
        --max_pending_prompts     1024 \
        2>&1 | tee "${LOG}" &

    PID=$!
    SERVER_PIDS+=("$PID")
    SERVER_URLS+=("http://${SERVER_HOST}:${PORT}")
    echo ">>>   Server ${i}: GPU=${GPU}  port=${PORT}  PID=${PID}  log=${LOG}"
done

# ── Step 2: 等待所有 server 就绪 ─────────────────────────────
# ClusterTopology.wait_until_ready() 在 trainer 内部也会等待，
# 这里在 shell 层提前轮询是一道额外保险，避免训练进程启动后
# 长时间看不到任何输出让人误以为卡死。
echo ">>> 等待所有 RolloutServer 就绪（最多 180s）..."

for i in "${!ROLLOUT_GPUS[@]}"; do
    PORT=$((BASE_PORT + i))
    for attempt in $(seq 1 90); do
        if curl -sf "http://${SERVER_HOST}:${PORT}/health" > /dev/null 2>&1; then
            echo ">>>   port ${PORT} 已就绪 (attempt=${attempt})"
            break
        fi
        sleep 2
        if [ "$attempt" -eq 90 ]; then
            echo ">>> 错误：port ${PORT} 启动超时"
            cleanup
        fi
    done
done

echo ">>> 所有 ${NUM_ROLLOUT_SERVERS} 个 RolloutServer 已就绪！"

# ── 构建 server_urls JSON 列表（传给 Hydra）──────────────────
# 格式: ["http://127.0.0.1:8000","http://127.0.0.1:8001",...]
# 用 ++ 前缀覆盖，避免 Hydra 解析 JSON 数组时出现引号转义问题。
URLS_JSON="["
for i in "${!SERVER_URLS[@]}"; do
    [ "$i" -gt 0 ] && URLS_JSON+=","
    URLS_JSON+="\"${SERVER_URLS[$i]}\""
done
URLS_JSON+="]"
echo ">>> server_urls: ${URLS_JSON}"

# ── 训练 GPU 数量（= FSDP world size）────────────────────────
IFS=',' read -ra _TRAIN_GPU_ARR <<< "${TRAIN_GPUS}"
N_TRAIN_GPUS=${#_TRAIN_GPU_ARR[@]}
echo ">>> 训练 GPU 数量: ${N_TRAIN_GPUS} (${TRAIN_GPUS})"

# ── Step 3: 启动训练 ──────────────────────────────────────────
# 说明：
#   - server_urls 非空时，训练侧不启动本地推理引擎
#   - PromptProducer 后台线程完全自驱提交 prompt，Trainer 只管 pull/train/push
#   - dispatch_strategy=least_pending 让 Dispatcher 优先选 pending 数最少的 server
#   - producer_backoff: server busy (503) 时 Producer 退避的秒数，默认 2.0
#   - staleness_tolerance: 训练侧允许拉取的最大版本滞后，3 表示接受 wv >= current-3
echo ">>> 启动训练 on GPU ${TRAIN_GPUS}..."

CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.n=5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.health_check_interval=10.0 \
    worker.rollout.dispatch_strategy=least_pending \
    "worker.rollout.server_urls=${URLS_JSON}" \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_TRAIN_GPUS} \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=5 \
    trainer.val_freq=10 \
    trainer.staleness_tolerance=3 \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}"

TRAIN_EXIT=$?

# ── 清理 ─────────────────────────────────────────────────────
echo ">>> 训练结束（exit code: ${TRAIN_EXIT}），关闭所有 RolloutServer..."

for pid in "${SERVER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done
sleep 3
pkill -f "launch_server" 2>/dev/null || true
pkill -f "sglang"        2>/dev/null || true
sleep 2
pkill -9 -f "launch_server" 2>/dev/null || true
pkill -9 -f "sglang"        2>/dev/null || true

REMAINING=$(ps aux | grep -E "launch_server|sglang" | grep -v grep | wc -l)
if [ "${REMAINING}" -eq 0 ]; then
    echo ">>> 所有进程已清理完毕"
else
    echo ">>> 警告：仍有 ${REMAINING} 个相关进程，请手动检查"
    ps aux | grep -E "launch_server|sglang" | grep -v grep
fi

exit ${TRAIN_EXIT}