import asyncio
import time
import torch
import numpy as np
from tensordict import TensorDict

async def benchmark_generate_sequences(rollout, num_prompts=10):
    """
    分解 generate_sequences 各阶段耗时
    """
    # 构造假数据
    prompt_len =256
    vocab_size = 32000
    
    dummy_input_ids = torch.randint(0, vocab_size, (1, prompt_len))
    dummy_attention_mask = torch.ones(1, prompt_len, dtype=torch.long)
    dummy_position_ids = torch.arange(prompt_len).unsqueeze(0)
    dummy_raw_prompt_ids = np.array(
        [dummy_input_ids[0].numpy()], dtype=object
    )

    from verl.protocol import DataProto
    batch = TensorDict(
        {
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask,
            "position_ids": dummy_position_ids,
        },
        batch_size=1,
    )
    prompts = DataProto(
        batch=batch,
        non_tensor_batch={"raw_prompt_ids": dummy_raw_prompt_ids},
        meta_info={
            "eos_token_id": 2,
            "temperature": 1.0,
        },
    )

    timings = {
        "preprocess": [],
        "sglang_generate": [],
        "postprocess": [],
        "total": [],
    }

    n = rollout.sampling_params.get("n", 1)

    for i in range(num_prompts):
        # 每次重新构造（模拟真实调用）
        import copy
        p = copy.deepcopy(prompts)
        input_ids = p.batch["input_ids"]
        attention_mask = p.batch["attention_mask"]
        position_ids = p.batch["position_ids"]
        eos_token_id = p.meta_info["eos_token_id"]
        batch_raw_prompt_ids = p.non_tensor_batch.pop("raw_prompt_ids")

        t_total_start = time.perf_counter()

        # ── 阶段1: 前处理 ──
        t0 = time.perf_counter()
        sglang_inputs = [
            {"input_ids": list(raw_prompt_ids)}
            for raw_prompt_ids in batch_raw_prompt_ids
        ]
        sampling_params = dict(rollout.sampling_params)
        t_preprocess = time.perf_counter() - t0

        # ── 阶段2: SGLang 生成 ──
        t0 = time.perf_counter()
        raw_out = await rollout.inference_engine.async_generate(
            input_ids=sglang_inputs[0]["input_ids"],
            sampling_params=sampling_params,
            return_logprob=False,
            stream=False,
            image_data=None,
        )
        t_sglang = time.perf_counter() - t0

        # ── 阶段3: 后处理 ──
        from verl.workers.rollout.sglang_rollout_areal import _get_field, _strip_prompt_overlap

        t0 = time.perf_counter()
        # flatten outputs

        if isinstance(raw_out, list):
            # 这个版本 SGLang 直接返回 list，每个元素是一个 sample
            out_ids_list = []
            for item in raw_out:
                if isinstance(item, dict):
                    out_ids_list.append(list(item["output_ids"]))
                elif hasattr(item, "output_ids"):
                    out_ids_list.append(list(item.output_ids))
                else:
                    print(f"item type={type(item)}, item={item}")
                    raise RuntimeError(f"Cannot parse list item: {type(item)}")
            print(f"via list: {len(out_ids_list)} samples")

        prompt_ids = list(batch_raw_prompt_ids[0])
        response_ids_list = [
            _strip_prompt_overlap(ids, prompt_ids) for ids in out_ids_list
        ]
        from verl.utils import torch_functional as VF
        response_ids = VF.pad_2d_list_to_length(
            response_ids_list, rollout.pad_token_id,
            max_length=rollout.config.response_length,
        ).to(input_ids.device)

        if n > 1:
            input_ids = input_ids.repeat_interleave(n, dim=0)
            attention_mask = attention_mask.repeat_interleave(n, dim=0)
            position_ids = position_ids.repeat_interleave(n, dim=0)

        batch_size = n
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta = torch.arange(1, response_length + 1).view(1, -1).expand(batch_size, -1)
        response_position_ids = position_ids[..., -1:] + delta
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        t_postprocess = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        timings["preprocess"].append(t_preprocess * 1000)
        timings["sglang_generate"].append(t_sglang * 1000)
        timings["postprocess"].append(t_postprocess * 1000)
        timings["total"].append(t_total * 1000)

        print(
            f"[{i+1:2d}/{num_prompts}] "
            f"pre={t_preprocess*1000:.2f}ms  "
            f"sglang={t_sglang*1000:.1f}ms  "
            f"post={t_postprocess*1000:.2f}ms  "
            f"total={t_total*1000:.1f}ms"
        )

    # 汇总
    print("\n" + "="*60)
    print(f"{'阶段':<20} {'均值':>10} {'最大':>10} {'占比':>10}")
    print("-"*60)
    avg_total = sum(timings["total"]) / len(timings["total"])
    for key in ["preprocess", "sglang_generate", "postprocess"]:
        avg = sum(timings[key]) / len(timings[key])
        mx = max(timings[key])
        ratio = avg / avg_total * 100
        print(f"{key:<20} {avg:>9.2f}ms {mx:>9.2f}ms {ratio:>9.1f}%")
    print("-"*60)
    print(f"{'total':<20} {avg_total:>9.2f}ms")
    print("="*60)


if __name__ == "__main__":
    # 用法：在 server 启动后单独跑这个脚本
    import sys
    sys.path.insert(0, "/home/ubuntu/qgz/EasyR1-async")

    from verl.workers.rollout.config import RolloutConfig
    from verl.utils.tokenizer import get_tokenizer
    from verl.workers.rollout.sglang_rollout_areal import SGLangRollout

    MODEL_PATH = "/home/ubuntu/huggingface/Qwen3-0.6B"

    tokenizer = get_tokenizer(MODEL_PATH, use_fast=True)
    config = RolloutConfig(
        name="vllm",
        n=5,
        temperature=1.0,
        dtype="bf16",
        gpu_memory_utilization=0.7,
        tensor_parallel_size=1,
        max_num_batched_tokens=8192,
    )
    config.prompt_length = 512
    config.response_length = 2048

    rollout = SGLangRollout(
        model_path=MODEL_PATH,
        config=config,
        tokenizer=tokenizer,
        processor=None,
    )

    asyncio.run(benchmark_generate_sequences(rollout, num_prompts=20))