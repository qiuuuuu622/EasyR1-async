import inspect
import re
import time
from dataclasses import asdict
from typing import Iterable, Union

import torch
import torch.distributed as dist
from peft import PeftModel, get_peft_model_state_dict
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
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
        module: torch.nn.Module,   # fully_shard()-applied model (FSDP2)
        inference_engine: LLM,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False

        # FSDP2: detect LoRA via the unwrapped module attribute
        # (fully_shard does not wrap in a new class; the model itself is mutated)
        self.is_lora = isinstance(module, PeftModel) or isinstance(
            getattr(module, "_orig_mod", None), PeftModel
        )

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

        # [OPT-1] Lazy-compiled key-renaming patterns (same optimisation as FSDP1 version)
        self._compiled_key_mapping: dict | None = None

    # ------------------------------------------------------------------
    # Key renaming (checkpoint_conversion_mapping)
    # ------------------------------------------------------------------

    def _rename_weight_keys(
        self,
        actor_weights: dict[str, Union[torch.Tensor, DTensor]],
        model: PreTrainedModel,
    ) -> dict[str, Union[torch.Tensor, DTensor]]:
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        if self._compiled_key_mapping is None:
            reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
            self._compiled_key_mapping = {}
            for pattern, replacement in reverse_key_mapping.items():
                replacement_clean = replacement.lstrip("^")
                replacement_clean = re.sub(r"\(.*\)", "", replacement_clean)
                self._compiled_key_mapping[re.compile(pattern)] = replacement_clean

        original_weights = {}
        for key, value in actor_weights.items():
            for compiled_pattern, replacement in self._compiled_key_mapping.items():
                key, n_replace = compiled_pattern.subn(replacement, key)
                if n_replace > 0:
                    break
            original_weights[key] = value

        return original_weights

    # ------------------------------------------------------------------
    # LoRA weight collection
    # ------------------------------------------------------------------

    def _collect_lora_weights(self) -> dict:
        """Collect LoRA weights from each transformer layer (FSDP2 edition).

        FSDP2 does not expose _all_handles / flat_param.  We iterate
        named_modules() to find transformer layers and use
        get_peft_model_state_dict to extract only the LoRA deltas.
        """
        lora_weights = {}
        peft_model = self.module  # fully_shard() mutates in-place

        # StateDictOptions: full_state_dict=True gathers DTensors to rank-0
        sd_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)

        for name, submodule in self.module.named_modules():
            layer_suffix = name.rsplit("layers.", 1)[-1]
            if not layer_suffix.isdigit():
                continue

            if self.use_param_offload:
                load_fsdp_submodule(submodule)

            # Gather the full state dict for this layer
            layer_sd = get_model_state_dict(submodule, options=sd_opts)

            # Extract LoRA deltas
            layer_lora_weights = get_peft_model_state_dict(peft_model, state_dict=layer_sd)
            for lora_module_name, lora_weight in layer_lora_weights.items():
                key = f"{name}.{lora_module_name}"
                if isinstance(lora_weight, DTensor):
                    lora_weights[key] = lora_weight.full_tensor().detach().cpu()
                else:
                    lora_weights[key] = lora_weight.detach().cpu()

            if self.use_param_offload:
                offload_fsdp_submodule(submodule)

            torch.cuda.empty_cache()

        return lora_weights

    # ------------------------------------------------------------------
    # Weight sync to vLLM
    # ------------------------------------------------------------------

    def _sync_weight_to_vllm(self) -> None:
        """Gather FSDP2-sharded weights and load them into the vLLM model."""

        # FSDP2: get_model_state_dict with full_state_dict=True gathers all
        # DTensor shards into contiguous CPU tensors on every rank by default
        # (or only on rank-0 when cpu_offload=True is paired with broadcast).
        sd_opts = StateDictOptions(full_state_dict=True, cpu_offload=False)

        if self.is_lora:
            actor_weights = self._collect_lora_weights()
        else:
            actor_weights = get_model_state_dict(self.module, options=sd_opts)
            # FSDP2 returns plain Tensors (not DTensors) when full_state_dict=True
            actor_weights = self._rename_weight_keys(actor_weights, self.module)

        vllm_model = (
            self.inference_engine.llm_engine
            .model_executor
            .driver_worker
            .worker
            .model_runner
            .model
        )

        if not self.is_lora:
            full_sd = {
                k: (v.full_tensor().cuda() if isinstance(v, DTensor) else v.cuda())
                for k, v in actor_weights.items()
            }
            vllm_model.load_state_dict(full_sd, strict=False)
            del full_sd
        else:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_request = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(self.module.peft_config.get("default")),
                lora_tensors=actor_weights,
            )
            self.inference_engine.llm_engine.add_lora(lora_request)

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Public lifecycle methods
    # ------------------------------------------------------------------

    def load_vllm_and_sync_weights(self) -> None:
        """Wake up the vLLM engine and sync model weights."""
        torch.cuda.empty_cache()
        assert not self.loaded, "vLLM engine has already been loaded"
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

    def offload_vllm(self) -> None:
        """Put the vLLM engine to sleep and reclaim GPU memory."""
        assert self.loaded, "vLLM engine has not been loaded"
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

    # ------------------------------------------------------------------
    # Data pre/post-processing (unchanged)
    # ------------------------------------------------------------------

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All-gather across the TP group so every rank sees identical input."""
        all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Keep only the chunk belonging to this TP rank."""
        if self.tp_size > 1:
            data = data.chunk(chunks=self.tp_size)[self.tp_rank]
        return data