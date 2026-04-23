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
Entry point for starting the standalone async RolloutServer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m verl.workers.rollout.launch_server \
        --model_path /path/to/model \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.9 \
        --pool_max_samples 5000 \
        --max_concurrent_prompts 256 \
        --prompt_length 2048 \
        --response_length 4096

The server must be started BEFORE launching the training job.
Training workers will:
  1. Push initial weights via /update_weights
  2. Submit prompts via /submit_prompts (non-blocking)
  3. Pull completed experiences via /pull_samples (blocking)
  4. Push updated weights after each training step
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Start the EasyR1 Async RolloutServer")
    parser.add_argument("--model_path", type=str, default='', help="Path to the model (HuggingFace format)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="SGLang tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="SGLang GPU memory utilization")
    parser.add_argument("--dtype", type=str, default="bf16", help="Model dtype (bf16, fp16, fp32)")
    parser.add_argument("--prompt_length", type=int, default=2048, help="Max prompt length")
    parser.add_argument("--response_length", type=int, default=512, help="Max response length")
    parser.add_argument("--max_model_len", type=int, default=None, help="Override max model len")
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192, help="Max batched tokens for SGLang")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--n", type=int, default=1, help="Number of rollout samples per prompt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_log_stats", action="store_true", default=True)

    # 经验池参数
    parser.add_argument(
        "--pool_max_samples", type=int, default=5000,
        help="Max number of samples in the experience pool",
    )
    parser.add_argument(
        "--max_pending_prompts", type=int, default=3,
        help="Max number of batchs in the pending queue",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import os

    from ..rollout.config import RolloutConfig
    from ...utils.tokenizer import get_processor, get_tokenizer

    rollout_config = RolloutConfig(
        name="vllm",
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        disable_log_stats=args.disable_log_stats,
        enforce_eager=args.enforce_eager,
    )
    rollout_config.prompt_length = args.prompt_length
    rollout_config.response_length = args.response_length
    rollout_config.trust_remote_code = args.trust_remote_code

    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = get_tokenizer(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    processor = get_processor(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    from .rollout_server import RolloutServer
    server = RolloutServer(
        model_path=args.model_path,
        rollout_config=rollout_config,
        tokenizer=tokenizer,
        processor=processor,
        host=args.host,
        port=args.port,
        pool_max_samples=args.pool_max_samples,
        max_pending_prompts=args.max_pending_prompts,
    )
    server.run()


if __name__ == "__main__":
    main()