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

import json
import signal
import sys
import time

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training with optional timeout control."""

    def run(self, config: PPOConfig):
        start_time = time.time()
        max_duration = 2 * 3600 - 300  # 2小时减5分钟

        print("====================================================")
        print(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"预设结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time + max_duration))}")
        print("====================================================")

        print(json.dumps(config.to_dict(), indent=2))

        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
        }

        global_pool_id = "global_pool"
        # n_gpus_per_node 决定训练用几张卡，在 bash 脚本里通过命令行覆盖
        n_gpus = config.trainer.n_gpus_per_node
        resource_pool_spec = {global_pool_id: [n_gpus]}
        mapping = {Role.ActorRolloutRef: global_pool_id}

        # GAE 算法需要 Critic
        if config.algorithm.adv_estimator == "gae":
            role_worker_mapping[Role.Critic] = ray.remote(FSDPWorker)
            mapping[Role.Critic] = global_pool_id

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        RemoteRewardManager = ray.remote(AutoRewardManager).options(
            num_cpus=config.worker.reward.num_cpus
        )
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        def signal_handler(sig, frame):
            print("\n接收到系统信号，尝试安全退出...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        trainer.init_workers()

        try:
            trainer.fit()
        except Exception as e:
            print(f"训练中断: {e}")
            raise
        finally:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass
            print("训练结束。")


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    try:
        ray.get(runner.run.remote(ppo_config), timeout=7200)
    except ray.exceptions.GetTimeoutError:
        print("\n[警告] 任务已运行满 2 小时，强制终止。")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        if ppo_config.trainer.ray_timeline is not None:
            try:
                ray.timeline(filename=ppo_config.trainer.ray_timeline)
            except Exception:
                pass
        if ray.is_initialized():
            ray.shutdown()
        print("Ray 已关闭，程序退出。")


if __name__ == "__main__":
    main()