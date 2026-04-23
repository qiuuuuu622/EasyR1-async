import asyncio
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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

try:
    import sglang as sgl
except ImportError:
    sgl = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profiling 数据结构
# ---------------------------------------------------------------------------

@dataclass
class PromptCompletionEvent:
    """单条 prompt 的完成记录"""
    prompt_idx: int              # 在本次 batch 中的原始下标
    finish_ts: float             # 完成时的绝对时间戳
    elapsed_s: float             # 从 batch 开始到完成的耗时（秒）
    num_tokens: int              # 生成的 token 数
    finish_reason: Optional[str] # "eos" / "length" / None
    order: int                   # 第几个完成（1-indexed）


@dataclass
class BatchRolloutProfile:
    """一个 batch 完整的完成时间线"""
    batch_size: int
    batch_start_ts: float
    batch_end_ts: float
    total_elapsed_s: float
    events: list[PromptCompletionEvent] = field(default_factory=list)

    # 派生统计
    p50_elapsed_s: float = 0.0
    p90_elapsed_s: float = 0.0
    p99_elapsed_s: float = 0.0
    max_elapsed_s: float = 0.0
    min_elapsed_s: float = 0.0
    mean_elapsed_s: float = 0.0

    # 长尾分析：最后 10% 完成的请求拖慢了多少
    tail_wait_s: float = 0.0        # p90 到 max 的差值
    tail_wait_ratio: float = 0.0    # tail_wait_s / total_elapsed_s

    def compute_stats(self):
        if not self.events:
            return
        elapsed_list = sorted(e.elapsed_s for e in self.events)
        n = len(elapsed_list)
        self.min_elapsed_s  = elapsed_list[0]
        self.max_elapsed_s  = elapsed_list[-1]
        self.mean_elapsed_s = sum(elapsed_list) / n
        self.p50_elapsed_s  = elapsed_list[int(n * 0.50)]
        self.p90_elapsed_s  = elapsed_list[int(n * 0.90)]
        self.p99_elapsed_s  = elapsed_list[min(int(n * 0.99), n - 1)]
        self.tail_wait_s    = self.max_elapsed_s - self.p90_elapsed_s
        self.tail_wait_ratio = (
            self.tail_wait_s / self.total_elapsed_s if self.total_elapsed_s > 0 else 0.0
        )

    def log_summary(self, step: Optional[int] = None):
        tag = f"[RolloutProfile step={step}]" if step is not None else "[RolloutProfile]"
        logger.info(
            f"{tag} batch_size={self.batch_size} "
            f"total={self.total_elapsed_s:.2f}s "
            f"min={self.min_elapsed_s:.2f}s "
            f"p50={self.p50_elapsed_s:.2f}s "
            f"p90={self.p90_elapsed_s:.2f}s "
            f"p99={self.p99_elapsed_s:.2f}s "
            f"max={self.max_elapsed_s:.2f}s | "
            f"tail_wait={self.tail_wait_s:.2f}s "
            f"tail_ratio={self.tail_wait_ratio:.1%}"
        )

    def log_timeline(self, step: Optional[int] = None, top_slow: int = 5):
        """打印完成顺序时间线，以及最慢的若干条 prompt"""
        tag = f"[RolloutProfile step={step}]" if step is not None else "[RolloutProfile]"

        # 完成顺序时间线（每隔 batch_size//10 打一条，防止刷屏）
        stride = max(1, self.batch_size // 10)
        lines = []
        for e in self.events:
            if e.order % stride == 0 or e.order == 1 or e.order == self.batch_size:
                bar_len = int(e.elapsed_s / self.total_elapsed_s * 40) if self.total_elapsed_s > 0 else 0
                bar = "█" * bar_len
                lines.append(
                    f"  order={e.order:>4d} prompt={e.prompt_idx:>4d} "
                    f"t={e.elapsed_s:>6.2f}s tokens={e.num_tokens:>5d} "
                    f"reason={e.finish_reason or '?':>6s} |{bar}"
                )
        logger.info(f"{tag} completion timeline (stride={stride}):\n" + "\n".join(lines))

        # 最慢的 top_slow 条
        slowest = sorted(self.events, key=lambda e: e.elapsed_s, reverse=True)[:top_slow]
        slow_lines = [
            f"  rank={i+1} prompt={e.prompt_idx} "
            f"elapsed={e.elapsed_s:.2f}s tokens={e.num_tokens} reason={e.finish_reason}"
            for i, e in enumerate(slowest)
        ]
        logger.info(f"{tag} slowest {top_slow} prompts:\n" + "\n".join(slow_lines))

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "total_elapsed_s": self.total_elapsed_s,
            "min_elapsed_s": self.min_elapsed_s,
            "p50_elapsed_s": self.p50_elapsed_s,
            "p90_elapsed_s": self.p90_elapsed_s,
            "p99_elapsed_s": self.p99_elapsed_s,
            "max_elapsed_s": self.max_elapsed_s,
            "mean_elapsed_s": self.mean_elapsed_s,
            "tail_wait_s": self.tail_wait_s,
            "tail_wait_ratio": self.tail_wait_ratio,
            "events": [
                {
                    "prompt_idx": e.prompt_idx,
                    "elapsed_s": round(e.elapsed_s, 4),
                    "num_tokens": e.num_tokens,
                    "finish_reason": e.finish_reason,
                    "order": e.order,
                }
                for e in self.events
            ],
        }


# ---------------------------------------------------------------------------
# 工具函数（与原版相同，保持不变）
# ---------------------------------------------------------------------------

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {int(image_token_id): -100.0}
    return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int
) -> Optional[dict[str, Any]]:
    if not multi_modal_data:
        return None
    images = []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))
    if images:
        return {"image_data": images}
    if "videos" in multi_modal_data:
        raise NotImplementedError("SGLang rollout currently supports images only.")
    return None


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
    if not output_ids or not prompt_ids:
        return output_ids
    max_overlap = min(len(output_ids), len(prompt_ids))
    for overlap in range(max_overlap, 0, -1):
        if prompt_ids[-overlap:] == output_ids[:overlap]:
            return output_ids[overlap:]
    return output_ids


def _parse_output_token_logprobs(meta_info: dict[str, Any], token_ids: list[int]) -> list[float]:
    raw = meta_info.get("output_token_logprobs")
    if raw is None:
        raise RuntimeError("SGLang did not return output_token_logprobs; set return_logprob=True.")
    parsed: list[float] = []
    for i, item in enumerate(raw[: len(token_ids)]):
        if isinstance(item, (float, int)):
            parsed.append(float(item))
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            parsed.append(0.0 if item[0] is None else float(item[0]))
        elif isinstance(item, dict):
            value = item.get("logprob", item.get("value", 0.0))
            parsed.append(0.0 if value is None else float(value))
        else:
            parsed.append(0.0)
    if len(parsed) < len(token_ids):
        parsed.extend([0.0] * (len(token_ids) - len(parsed)))
    return parsed


def _get_finish_reason(out: Any) -> Optional[str]:
    """兼容多版本 SGLang 取 finish_reason"""
    reason = _get_field(out, "finish_reason")
    if reason is not None:
        return str(reason)
    meta = _get_field(out, "meta_info")
    if meta is not None:
        return str(_get_field(meta, "finish_reason", "unknown"))
    return None


# ---------------------------------------------------------------------------
# SGLangRollout（profiling 版）
# ---------------------------------------------------------------------------

class SGLangRollout(BaseRollout):

    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        server_mode: bool = False,
        # ---- profiling 开关 ----
        enable_rollout_profile: bool = True,
        profile_log_timeline: bool = True,
        profile_top_slow: int = 5,
        **kwargs,
    ):
        super().__init__()
        if sgl is None:
            raise ImportError("sglang is not installed.")

        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.server_mode = server_mode
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.processor = processor

        # profiling 配置
        self.enable_rollout_profile = enable_rollout_profile
        self.profile_log_timeline = profile_log_timeline
        self.profile_top_slow = profile_top_slow
        self._profile_step: int = 0           # 自动递增的 step 计数
        self._profile_history: list[BatchRolloutProfile] = []  # 保留最近 N 个 batch 的 profile

        if torch.distributed.is_initialized():
            if config.tensor_parallel_size > torch.distributed.get_world_size():
                raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        self.lora_kwargs = kwargs.pop("lora_kwargs", {})
        if self.lora_kwargs:
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
        }
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        self.inference_engine = sgl.Engine(**engine_kwargs)

        sampling_kwargs = {
            "max_new_tokens": config.response_length,
            "n": getattr(config, "n", 1),
        }
        mapped_keys = [
            "temperature", "top_p", "top_k", "min_p",
            "presence_penalty", "frequency_penalty", "repetition_penalty",
            "stop", "stop_token_ids", "ignore_eos",
            "skip_special_tokens", "no_stop_trim",
        ]
        for key in mapped_keys:
            if hasattr(config, key):
                sampling_kwargs[key] = getattr(config, key)

        self.sampling_params = sampling_kwargs
        logger.info(f"SGLang sampling params: {self.sampling_params}")

    # ------------------------------------------------------------------
    # update_sampling_params（与原版相同）
    # ------------------------------------------------------------------

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                mapped_key = "max_new_tokens" if key == "max_tokens" else key
                if mapped_key in self.sampling_params or mapped_key in {
                    "temperature", "top_p", "top_k", "min_p",
                    "presence_penalty", "frequency_penalty", "repetition_penalty",
                    "stop", "stop_token_ids", "ignore_eos",
                    "skip_special_tokens", "no_stop_trim", "n", "max_new_tokens",
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

    # ------------------------------------------------------------------
    # 核心：逐请求提交 + 收集完成时间戳
    # ------------------------------------------------------------------

    async def _generate_single(
        self,
        prompt_idx: int,
        input_ids_list: list[int],
        sampling_params: dict,
        image_data: Optional[list] = None,
    ) -> tuple[int, list[Any], float]:
        """
        提交单条请求给 SGLang，返回 (prompt_idx, flat_outputs, finish_ts)。
        flat_outputs 已经展开为 n 个独立 output 对象，与原版 n>1 处理对齐。
        所有请求在同一个 event loop 里并发提交，SGLang 内部依然走 continuous batching。
        """
        kwargs = dict(
            input_ids=[input_ids_list],
            sampling_params=[sampling_params],
            return_logprob=False,
            stream=False,
        )
        if image_data is not None:
            kwargs["image_data"] = [image_data]

        raw = await self.inference_engine.async_generate(**kwargs)
        finish_ts = time.perf_counter()

        raw = _maybe_to_list(raw)
        # raw 是一个列表，对于单条请求：
        #   n=1 → raw = [output]，直接取 raw[0]
        #   n>1 → SGLang 可能返回：
        #     a) raw = [out0, out1, ..., out_{n-1}]  已展开（len==n）
        #     b) raw = [out]，out 内有 output_ids_list / outputs 字段（未展开）
        n = sampling_params.get("n", 1)
        flat: list[Any] = []

        if len(raw) == n:
            # 情况 a：SGLang 已经展开成 n 条
            flat = raw
        else:
            # 情况 b：一条 output 里包含 n 个结果，需要手动展开
            out = raw[0] if raw else None
            if out is None:
                flat = [None] * n
            elif hasattr(out, "output_ids_list"):
                # 新版 SGLang：output_ids_list = [ids_0, ids_1, ..., ids_{n-1}]
                for ids in out.output_ids_list:
                    flat.append({"output_ids": ids})
            elif hasattr(out, "outputs"):
                # 部分版本：outputs = [CompletionOutput_0, ..., CompletionOutput_{n-1}]
                flat.extend(out.outputs)
            else:
                # 未知结构：打印诊断信息，帮助定位实际字段
                logger.warning(
                    f"[_generate_single] prompt_idx={prompt_idx} n={n} "
                    f"len(raw)={len(raw)} unknown output structure. "
                    f"type={type(out)} "
                    f"attrs={[a for a in dir(out) if not a.startswith('__')]}"
                )
                # 尝试直接从 output 对象取 output_ids（n=1 退化情况）
                out_ids = _get_field(out, "output_ids")
                if out_ids is not None and n == 1:
                    flat = [out]
                else:
                    # 真正兜底：复制 n 份，下游会报 mismatch，但至少给出诊断信息
                    flat = raw * n if raw else [None] * n

        if len(flat) != n:
            logger.error(
                f"[_generate_single] prompt_idx={prompt_idx} "
                f"flat展开后长度={len(flat)} 期望n={n}，展开有误"
            )

        return prompt_idx, flat, finish_ts

    async def _generate_with_profile(
        self,
        sglang_inputs: list[dict],
        per_request_sampling_params: list[dict],
        n: int,
    ) -> tuple[list[Any], Optional[BatchRolloutProfile]]:
        """
        并发提交所有请求，用 asyncio.as_completed 收集完成顺序和时间戳。
        返回：
          - outputs: 已展开为 batch_size * n 条，与原版 async_generate 输出对齐
          - profile: BatchRolloutProfile（若 enable_rollout_profile=False 则为 None）
        """
        batch_size = len(sglang_inputs)
        batch_start = time.perf_counter()

        # 用 loop.create_task 而不是 ensure_future，确保绑定到当前 running loop
        # （_generate_with_profile 是 async def，一定在 main_loop 里被 await，所以安全）
        loop = asyncio.get_running_loop()
        tasks = {
            loop.create_task(
                self._generate_single(
                    prompt_idx=i,
                    input_ids_list=sglang_inputs[i]["input_ids"],
                    sampling_params=per_request_sampling_params[i],
                    image_data=sglang_inputs[i].get("image_data"),
                )
            ): i
            for i in range(batch_size)
        }

        # 按完成顺序收集结果
        # outputs_by_idx[prompt_idx] = list of n flat outputs
        outputs_by_idx: dict[int, list[Any]] = {}
        events: list[PromptCompletionEvent] = []
        order = 0

        for coro in asyncio.as_completed(list(tasks.keys())):
            prompt_idx, flat_outputs, finish_ts = await coro
            elapsed = finish_ts - batch_start
            order += 1

            outputs_by_idx[prompt_idx] = flat_outputs

            if self.enable_rollout_profile:
                # profiling 用第一个 sample 的 token 数代表这条 prompt 的完成情况
                first = flat_outputs[0] if flat_outputs else None
                out_ids = _get_field(first, "output_ids") if first is not None else []
                num_tokens = len(out_ids) if out_ids else 0
                events.append(PromptCompletionEvent(
                    prompt_idx=prompt_idx,
                    finish_ts=finish_ts,
                    elapsed_s=elapsed,
                    num_tokens=num_tokens,
                    finish_reason=_get_finish_reason(first) if first else None,
                    order=order,
                ))

        batch_end = time.perf_counter()
        total_elapsed = batch_end - batch_start

        # 按原始 prompt 顺序重组，展开为 batch_size * n 条
        # 顺序：prompt_0的n条, prompt_1的n条, ...（与原版 repeat_interleave 对齐）
        outputs: list[Any] = []
        for i in range(batch_size):
            outputs.extend(outputs_by_idx[i])

        # 构建 profile
        profile = None
        if self.enable_rollout_profile:
            profile = BatchRolloutProfile(
                batch_size=batch_size,
                batch_start_ts=batch_start,
                batch_end_ts=batch_end,
                total_elapsed_s=total_elapsed,
                events=sorted(events, key=lambda e: e.order),
            )
            profile.compute_stats()

        return outputs, profile

    # ------------------------------------------------------------------
    # generate_sequences（profiling 版）
    # ------------------------------------------------------------------

    @torch.no_grad()
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor       = prompts.batch["input_ids"]
        attention_mask: torch.Tensor  = prompts.batch["attention_mask"]
        position_ids: torch.Tensor    = prompts.batch["position_ids"]
        eos_token_id: int             = prompts.meta_info["eos_token_id"]

        if "temperature" not in prompts.meta_info:
            prompts.meta_info["temperature"] = self.config.temperature

        batch_size = input_ids.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids   = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("sglang sharding manager is not working properly.")

        # 构建 sglang_inputs（与原版相同）
        if batch_multi_modal_data is not None:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                item = {"input_ids": list(raw_prompt_ids)}
                item.update(
                    _process_multi_modal_data(
                        multi_modal_data,
                        prompts.meta_info["min_pixels"],
                        prompts.meta_info["max_pixels"],
                    ) or {}
                )
                sglang_inputs.append(item)
        else:
            sglang_inputs = [{"input_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        with self.update_sampling_params(**prompts.meta_info):
            n = int(self.sampling_params.get("n", 1))
            per_request_sampling_params = [dict(self.sampling_params) for _ in range(batch_size)]

            # -------------------------------------------------------
            # 核心：并发生成 + profiling
            # -------------------------------------------------------
            outputs, profile = await self._generate_with_profile(
                sglang_inputs=sglang_inputs,
                per_request_sampling_params=per_request_sampling_params,
                n=n,
            )

            # 打印 profiling 结果（仅 rank 0）
            if profile is not None and self.rank == 0:
                step = self._profile_step
                profile.log_summary(step=step)
                if self.profile_log_timeline:
                    profile.log_timeline(step=step, top_slow=self.profile_top_slow)

                # 保留历史，方便后续聚合分析
                self._profile_history.append(profile)
                if len(self._profile_history) > 100:
                    self._profile_history.pop(0)

            self._profile_step += 1

            # -------------------------------------------------------
            # 以下与原版完全相同
            # -------------------------------------------------------
            expected = batch_size * n
            if len(outputs) != expected:
                logger.error(
                    f"[SGLangRollout] output size mismatch expected={expected} actual={len(outputs)}"
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
                    raise RuntimeError("SGLang output does not contain output_ids.")
                out_ids = _strip_prompt_overlap(list(out_ids), prompt_ids)
                response_ids_list.append(out_ids)

            response_ids = VF.pad_2d_list_to_length(
                response_ids_list, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if n > 1:
                batch_size      = batch_size * n
                input_ids       = _repeat_interleave(input_ids, n)
                attention_mask  = _repeat_interleave(attention_mask, n)
                position_ids    = _repeat_interleave(position_ids, n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, n)

        sequence_ids   = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(
                batch_size, 1, -1
            ).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids   = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask  = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts":        input_ids,
                "responses":      response_ids,
                "input_ids":      sequence_ids,
                "attention_mask": attention_mask,
                "response_mask":  response_mask,
                "position_ids":   position_ids,
            },
            batch_size=batch_size,
        )

        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    # ------------------------------------------------------------------
    # 聚合多个 batch 的 profiling 统计（供外部调用）
    # ------------------------------------------------------------------

    def get_profile_summary(self) -> Optional[dict]:
        """返回最近所有 batch 的聚合统计，用于判断长尾严重程度"""
        if not self._profile_history:
            return None

        all_tail_ratios  = [p.tail_wait_ratio  for p in self._profile_history]
        all_p90          = [p.p90_elapsed_s     for p in self._profile_history]
        all_max          = [p.max_elapsed_s     for p in self._profile_history]
        all_total        = [p.total_elapsed_s   for p in self._profile_history]

        return {
            "num_batches_profiled":  len(self._profile_history),
            "avg_total_elapsed_s":   sum(all_total) / len(all_total),
            "avg_p90_elapsed_s":     sum(all_p90) / len(all_p90),
            "avg_max_elapsed_s":     sum(all_max) / len(all_max),
            "avg_tail_wait_ratio":   sum(all_tail_ratios) / len(all_tail_ratios),
            "max_tail_wait_ratio":   max(all_tail_ratios),
            # 建议的 over-sampling 倍率：基于 p90/max 比值估算
            # 直觉：如果 p90=5s max=20s，说明打断点在 5s 能省掉 75% 等待
            "suggested_oversample_ratio": round(
                all_max[-1] / all_p90[-1] if all_p90[-1] > 0 else 1.5, 2
            ),
        }
        
        