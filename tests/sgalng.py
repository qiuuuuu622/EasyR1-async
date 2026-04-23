import os
import sglang as sgl

if __name__ == '__main__':
    model_path = os.path.expandvars("${HOME}/huggingface/Qwen3-0.6B")
    llm = sgl.Engine(model_path=model_path)

    input_ids = [[134, 244, 3098, 42, 235, 6, 37, 8]]
    prompt_len = len(input_ids[0])
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 10}

    outputs = llm.generate(
        input_ids=input_ids,
        sampling_params=sampling_params,
        return_logprob=True,
        top_logprobs_num=1,
        logprob_start_len=prompt_len - 1,
    )

    out = outputs[0]
    print("out keys:", list(out.keys()))

    # 找 output token ids
    text = out.get("text", "")
    meta_info = out["meta_info"]
    output_token_logprobs = meta_info["output_token_logprobs"]

    print("prompt_len:", prompt_len)
    print("text:", text)
    print("output_token_logprobs length:", len(output_token_logprobs))
    print("output_token_logprobs:", output_token_logprobs)

    # 期望 logprobs 长度 == 10（生成 token 数）
    # logprob_start_len=prompt_len-1 会多返回最后一个 prompt token 的 logprob
    # 所以实际长度应该是 10 + 1
    print("expected length: 10 or 11")

    token_logprobs = [
        float(item[0]) if isinstance(item, (list, tuple)) and item[0] is not None
        else 0.0
        for item in output_token_logprobs
    ]
    print("parsed token_logprobs:", token_logprobs)

    llm.shutdown()