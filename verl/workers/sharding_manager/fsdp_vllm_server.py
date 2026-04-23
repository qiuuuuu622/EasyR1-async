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
FSDPVLLMServerShardingManager — 异步解耦版本 (sample 粒度经验池)
"""

from ...protocol import DataProto
from .base import BaseShardingManager


class FSDPVLLMServerShardingManager(BaseShardingManager):

    def __init__(self, rollout_client, use_param_offload: bool = False):
        super().__init__()
        self.rollout_client = rollout_client
        self.use_param_offload = use_param_offload
        self.freed_bytes = 0
        
    def load_vllm_and_sync_weights(self, flush_cache: bool = False):
        """Push FSDP weights to server."""
        self.rollout_client.push_weights(use_param_offload=self.use_param_offload, flush_cache=flush_cache)

    def offload_vllm(self):
        """No-op: server vLLM always running."""
        pass

    def submit_prompts(self, new_batch: DataProto, gen_batch: DataProto):
        """提交 (new_batch, gen_batch) 到 server（非阻塞）"""
        self.rollout_client.submit_prompts(new_batch, gen_batch)

    def pull_samples(
        self,
        target_samples: int,
        min_weight_version: int = 0,
        timeout: float = 300.0,
    ) -> dict:
        """从 server 经验池拉取 sample（阻塞等待）"""
        return self.rollout_client.pull_samples(
            target_samples=target_samples,
            min_weight_version=min_weight_version,
            timeout=timeout,
        )

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
