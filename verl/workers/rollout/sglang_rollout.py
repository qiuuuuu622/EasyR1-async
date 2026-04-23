import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
import asyncio

try:
    import sglang as sgl
except ImportError:  # pragma: no cover
    sgl = None


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)



def _maybe_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _strip_prompt_overlap(output_ids: list[int], prompt_ids: list[int]) -> list[int]:
    """
    Defensive fix for SGLang versions where output_ids may contain an overlapping
    suffix of prompt_ids as a prefix of output_ids.
    """
    if not output_ids or not prompt_ids:
        return output_ids

    max_overlap = min(len(output_ids), len(prompt_ids))
    for overlap in range(max_overlap, 0, -1):
        if prompt_ids[-overlap:] == output_ids[:overlap]:
            return output_ids[overlap:]
    return output_ids


class SGLangRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        server_mode: bool = False,
        **kwargs,
    ):
        super().__init__()
        if sgl is None:
            raise ImportError("sglang is not installed. Please install sglang before using SGLangRollout.")

        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.server_mode = server_mode
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.processor = processor

        if torch.distributed.is_initialized():
            if config.tensor_parallel_size > torch.distributed.get_world_size():
                raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        self.lora_kwargs = kwargs.pop("lora_kwargs", {})
        if self.lora_kwargs:
            # You can wire this to per-request lora_path later if your workflow needs it.
            # Leaving it explicit avoids silent incompatibility.
            raise NotImplementedError("LoRA requests are not implemented for SGLangRollout yet.")

        engine_kwargs = {
            "model_path": model_path,
            "trust_remote_code": config.trust_remote_code,
            "dtype": PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            "context_length": config.max_model_len or (config.prompt_length + config.response_length),
            "tp_size": config.tensor_parallel_size,
            "mem_fraction_static": config.gpu_memory_utilization,
            "chunked_prefill_size": config.max_num_batched_tokens,
            "skip_tokenizer_init": True,
            "enable_cache_report": True, 
            "kv_cache_dtype": "fp8_e5m2",  # KV Cache用FP8压缩
            "enable_metrics": True,
            "enable_prefix_caching": True,
            # "quantization": "fp8",         # 权重FP8量化
            **(
                {
                    "speculative_draft_model_path": config.speculative_draft_model_path,
                    "speculative_num_steps": config.speculative_num_steps,
                    "speculative_num_draft_tokens": config.speculative_num_draft_tokens,
                }
                if config.speculative_draft_model_path
                else {}
            ), 
        }
        

        # Some SGLang versions do not accept all kwargs. Remove values that are None.
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}

        self.inference_engine = sgl.Engine(**engine_kwargs)

        sampling_kwargs = {
            "max_new_tokens": config.response_length,
            # logit_bias 不被 sglang SamplingParams 支持，已移除
            "n": getattr(config, "n", 1),
        }

        # Map common generation parameters from rollout config into SGLang sampling params.
        # Only pass parameters that are actually present on config.
        mapped_keys = [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
            "stop_token_ids",
            "ignore_eos",
            "skip_special_tokens",
            "no_stop_trim",
        ]
        for key in mapped_keys:
            if hasattr(config, key):
                sampling_kwargs[key] = getattr(config, key)

        self.sampling_params = sampling_kwargs
        print(f"SGLang sampling params: {self.sampling_params}.")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                mapped_key = "max_new_tokens" if key == "max_tokens" else key
                if mapped_key in self.sampling_params or mapped_key in {
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "presence_penalty",
                    "frequency_penalty",
                    "repetition_penalty",
                    "stop",
                    "stop_token_ids",
                    "ignore_eos",
                    "skip_special_tokens",
                    "no_stop_trim",
                    "n",
                    "max_new_tokens",
                    # logit_bias 不被 sglang 支持，已移除
                }:
                    old_sampling_params_args[mapped_key] = self.sampling_params.get(mapped_key, None)
                    self.sampling_params[mapped_key] = value
        try:
            yield
        finally:
            for key, value in old_sampling_params_args.items():
                if value is None and key in self.sampling_params:
                    self.sampling_params.pop(key, None)
                else:
                    self.sampling_params[key] = value


    @torch.no_grad()
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]

        if "temperature" not in prompts.meta_info:
            prompts.meta_info["temperature"] = self.config.temperature

        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("sglang sharding manager is not working properly.")

        sglang_inputs = []
        sglang_inputs = [{"input_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        with self.update_sampling_params(**prompts.meta_info):
            per_request_sampling_params = [dict(self.sampling_params) for _ in range(batch_size)]            
            outputs = await self.inference_engine.async_generate(
                input_ids=[item["input_ids"] for item in sglang_inputs],
                sampling_params=per_request_sampling_params,
                return_logprob=False,
                stream=False,
            )

            outputs = _maybe_to_list(outputs)
            n = int(self.sampling_params.get("n", 1))
            expected = batch_size * n

            if len(outputs) != expected:
                logger.error(
                    "[SGLangRollout] output size mismatch "
                    f"(expected={expected}, actual={len(outputs)})"
                )
                raise RuntimeError(
                    f"Unexpected number of SGLang outputs. Expected {expected}, got {len(outputs)}."
                )

            response_ids_list = []
            for idx, out in enumerate(outputs):
                prompt_idx = idx // n if n > 1 else idx
                prompt_ids = list(batch_raw_prompt_ids[prompt_idx])

                out_ids = _get_field(out, "output_ids")
                if out_ids is None:
                    raise RuntimeError("SGLang output does not contain output_ids. Ensure skip_tokenizer_init=True.")

                out_ids = _strip_prompt_overlap(list(out_ids), prompt_ids)
                response_ids_list.append(out_ids)

            response_ids = VF.pad_2d_list_to_length(
                response_ids_list, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)
            del response_ids_list
            
            
            if n > 1:
                batch_size = batch_size * n
                input_ids = _repeat_interleave(input_ids, n)
                attention_mask = _repeat_interleave(attention_mask, n)
                position_ids = _repeat_interleave(position_ids, n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)    
        