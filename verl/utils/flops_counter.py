from typing import TYPE_CHECKING, List, Tuple
import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig


# =========================
# 1️⃣ 设备 FLOPS（补全 4090）
# =========================
def get_device_flops(unit: str = "T") -> float:
    def unit_convert(number: float, level: str):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number

        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = torch.cuda.get_device_name()
    flops = float("inf")

    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "4090" in device_name:
        # ⚠️ 关键补充：4090 FP16（保守值）
        flops = 82e12
    else:
        print(f"[WARN] Unknown GPU type: {device_name}, using inf FLOPS")

    return unit_convert(flops, unit)


# =========================
# 2️⃣ FlopsCounter（改进版）
# =========================
class FlopsCounter:
    """
    改进点：
    - 支持 4090
    - 支持 train / infer
    - 支持 vLLM（prefill / decode）
    """

    def __init__(
        self,
        config: "PretrainedConfig",
        mode: str = "infer",  # "train" or "infer"
    ):
        self.config = getattr(config, "text_config", config)
        self.mode = mode

        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.vocab_size = self.config.vocab_size
        self.num_heads = self.config.num_attention_heads
        self.intermediate_size = self.config.intermediate_size

        self.num_params = self._estimate_params()


    def _estimate_params(self) -> float:
        h = self.hidden_size
        L = self.num_layers
        inter = self.intermediate_size
        attn = 4 * h * h
        mlp = 3 * h * inter

        return L * (attn + mlp)


    def _dense_flops(self, tokens: int) -> float:
        """
        train: forward + backward → 6x
        infer: forward only → 2x
        """
        factor = 6 if self.mode == "train" else 2
        return factor * self.num_params * tokens


    def _attention_flops(
        self,
        batch_seqlens: List[int],
        is_prefill: bool = True,
    ) -> float:
        """
        prefill: O(n^2)
        decode: O(n)
        """
        h = self.hidden_size
        L = self.num_layers

        total = 0
        for s in batch_seqlens:
            if self.mode == "train":
                total += s * s
            else:
                if is_prefill:
                    total += s * s
                else:
                    total += s

        return total * h * L * 2

    def estimate_flops(
        self,
        batch_seqlens: List[int],
        delta_time: float,
        is_prefill: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Args:
            batch_seqlens: 每个样本 token 数
            delta_time: 执行时间（秒）
            is_prefill: True=prompt阶段, False=decode阶段

        Returns:
            achieved_tflops
            promised_tflops
            mfu
        """

        tokens_sum = sum(batch_seqlens)

        dense_flops = self._dense_flops(tokens_sum)
        attn_flops = self._attention_flops(batch_seqlens, is_prefill)

        total_flops = dense_flops + attn_flops

        achieved = total_flops / delta_time / 1e12
        promised = get_device_flops()

        mfu = achieved / promised if promised > 0 else 0.0

        return achieved, promised, mfu