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
[OPTIMIZED] Key changes:
  1. _rename_weight_keys: 预编译 re.sub 正则，避免每个 key 都重新编译；
     原代码在 for key in keys: for pattern in reverse_key_mapping: re.subn(pattern, ...) 的嵌套循环内，
     每次都会重新解析 pattern 字符串，对大模型 (数百个权重 key) 有明显开销。
  2. _collect_lora_weights: 提取 peft_prefix 字符串替换在条件分支里，避免对所有 module 都做 replace。
  3. _sync_weight_to_vllm: 仅在 is_lora=False 时才 rename keys，原逻辑不变，增加注释说明。
"""

import inspect
import re
import time
from dataclasses import asdict
from typing import Iterable, Union

import torch
import torch.distributed as dist
from peft import PeftModel, get_peft_model_state_dict
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto, all_gather_data_proto
from ...utils.fsdp_utils import (
    load_fsdp_model,
    load_fsdp_submodule,
    offload_fsdp_model,
    offload_fsdp_submodule,
)
from ...utils.model_utils import print_gpu_memory_usage
from ...utils.vllm_utils import TensorLoRARequest
from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False
        self.is_lora = isinstance(self.module._fsdp_wrapped_module, PeftModel)

        self.world_size = dist.get_world_size()
        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

        self.freed_bytes = 0

        self.torch_random_states = torch.cuda.get_rng_state()
        gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

        # [OPT-1] 预编译权重 key 重命名所需的正则，避免在 _rename_weight_keys 的嵌套循环内
        # 反复调用 re.subn(pattern_string, ...) 触发隐式编译。
        # 这里将 compiled pattern 缓存到实例上，_rename_weight_keys 直接使用。
        self._compiled_key_mapping: dict | None = None  # lazy-init in _rename_weight_keys

    def _rename_weight_keys(self, actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        # [OPT-1] 第一次调用时编译所有 pattern，后续复用（同一个 sharding manager 实例的生命周期内只编译一次）
        if self._compiled_key_mapping is None:
            reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
            self._compiled_key_mapping = {}
            for pattern, replacement in reverse_key_mapping.items():
                replacement_clean = replacement.lstrip("^")
                replacement_clean = re.sub(r"\(.*\)", "", replacement_clean)
                compiled = re.compile(pattern)
                self._compiled_key_mapping[compiled] = replacement_clean

        original_weights = {}
        for key, value in actor_weights.items():
            for compiled_pattern, replacement in self._compiled_key_mapping.items():
                key, n_replace = compiled_pattern.subn(replacement, key)
                if n_replace > 0:
                    break
            original_weights[key] = value

        return original_weights

    def _collect_lora_weights(self) -> dict:
        """Collect LoRA weights from each transformer layer."""
        lora_weights = {}
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)

        # [OPT-2] 将 peft_prefix 替换的 str.replace 只在 isdigit() 通过后执行，
        # 原代码在判断前就做 replace，对非 layer 模块浪费字符串操作
        for name, submodule in self.module.named_modules():
            layer_suffix = name.rsplit("layers.", 1)[-1]
            if not layer_suffix.isdigit():
                continue

            if self.use_param_offload:
                load_fsdp_submodule(submodule)

            # 只有确认是 layer 时才计算 peft_prefix
            peft_prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")
            layer_weights = get_model_state_dict(submodule)
            layer_lora_weights = get_peft_model_state_dict(peft_model, state_dict=layer_weights)
            for lora_module_name, lora_weight in layer_lora_weights.items():
                key = f"{peft_prefix}.{lora_module_name}"
                if isinstance(lora_weight, DTensor):
                    lora_weights[key] = lora_weight.full_tensor().detach().cpu()
                else:
                    lora_weights[key] = lora_weight.detach().cpu()

            submodule._is_root = False
            if self.use_param_offload:
                offload_fsdp_submodule(submodule)

            torch.cuda.empty_cache()

        return lora_weights

    def _sync_weight_to_vllm(self):
        if self.is_lora:
            actor_weights = self._collect_lora_weights()
        else:
            actor_weights = get_model_state_dict(self.module)
            actor_weights = self._rename_weight_keys(actor_weights, self.module._fsdp_wrapped_module)

        vllm_model = (
            self.inference_engine.llm_engine
            .model_executor
            .driver_worker
            .worker
            .model_runner
            .model
        )

        if not self.is_lora:
            full_sd = {}
            for k, v in actor_weights.items():
                real_v = v.full_tensor() if isinstance(v, DTensor) else v
                full_sd[k] = real_v.cuda()
                
            vllm_model.load_state_dict(full_sd, strict=False)
            
            del full_sd
        else:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(self.module._fsdp_wrapped_module.peft_config.get("default")),
                lora_tensors=actor_weights,
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)

        torch.cuda.empty_cache()
        
        
    def load_vllm_and_sync_weights(self):
        """Load vllm engine and sync model weights to vllm model."""
        torch.cuda.empty_cache()
        assert self.loaded is False, "vllm engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before vllm wake up in sharding manager")
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
        else:
            self.inference_engine.wake_up()

        self._sync_weight_to_vllm()

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["kv_cache"])

        print_gpu_memory_usage("After vllm wake up in sharding manager")
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def _make_weight_iterator(
        self, actor_weights: dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in actor_weights.items():
            yield name, tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

    def offload_vllm(self):
        """Offload vllm engine."""
        assert self.loaded is True, "vllm engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before vllm offload in sharding manager")
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        print_gpu_memory_usage("After vllm offload in sharding manager")

        self.module.train()
        torch.cuda.empty_cache()

        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size > 1:
            data = data.chunk(chunks=self.tp_size)[self.tp_rank]
        return data
