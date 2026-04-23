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
RolloutClient — 异步解耦版本 (sample 粒度经验池)

关键方法:
  - push_weights(): 推送 FSDP 权重到 server
  - submit_prompts(new_batch, gen_batch): 提交 prompts（非阻塞）
  - pull_samples(target_samples, min_wv): 拉取指定数量的 sample（阻塞）
  - generate(): 同步生成（用于 validation，保持兼容）
"""

import logging
import pickle
import time

import requests
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto

logger = logging.getLogger(__name__)

WEIGHT_PUSH_TIMEOUT = 600
GENERATE_TIMEOUT = 600
SUBMIT_TIMEOUT = 60
PULL_TIMEOUT = 1800


class RolloutClient:
    """
    HTTP client for the async RolloutServer (sample-granularity pool).
    """

    def __init__(
        self,
        server_url: str,
        fsdp_module: FSDP,
        rank: int = 0,
    ):
        self.server_url = server_url.rstrip("/")
        self.fsdp_module = fsdp_module
        self.rank = rank
        self.weight_version = 0
        self.freed_bytes = 0

        if self.rank == 0:
            self._wait_for_server()

    def _wait_for_server(self, timeout: int = 120, interval: float = 2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.server_url}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info(f"RolloutServer is ready: {resp.json()}")
                    return
            except requests.exceptions.ConnectionError:
                pass
            logger.info(f"Waiting for RolloutServer at {self.server_url}...")
            time.sleep(interval)
        raise RuntimeError(f"RolloutServer did not become ready within {timeout}s")

    def push_weights(self, use_param_offload: bool = False, flush_cache : bool = False):
        """Gather FSDP weights and push to server. All ranks participate."""
        from ...utils.fsdp_utils import load_fsdp_model, offload_fsdp_model

        if use_param_offload:
            load_fsdp_model(self.fsdp_module)

        logger.info("Gathering FSDP model state dict for weight sync...")
        t0 = time.time()

        state_dict = get_model_state_dict(self.fsdp_module)

        # [FIX] 显存优化: 逐 tensor 转 CPU，不在显存中同时持有两份完整模型权重
        cpu_state_dict = {}
        for name, tensor in state_dict.items():
            if hasattr(tensor, "full_tensor"):
                cpu_state_dict[name] = tensor.full_tensor().cpu()
            else:
                cpu_state_dict[name] = tensor.cpu()
        del state_dict

        if use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self.rank == 0:
            self.weight_version += 1
            payload = {"weight_version": self.weight_version, "state_dict": cpu_state_dict, "flush_cache": flush_cache}
            payload_bytes = pickle.dumps(payload)
            # [FIX] 序列化后立即释放 dict，避免 CPU 内存中同时驻留 dict 和 bytes 两份
            del payload, cpu_state_dict

            logger.info(
                f"Pushing weights v{self.weight_version} "
                f"({len(payload_bytes) / 1024 / 1024:.1f} MB)..."
            )
            resp = requests.post(
                f"{self.server_url}/update_weights",
                data=payload_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=WEIGHT_PUSH_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info(f"Weight push response: {resp.json()}")
        else:
            del cpu_state_dict

        if dist.is_initialized():
            dist.barrier()

        elapsed = time.time() - t0
        logger.info(f"Weight sync total time: {elapsed:.2f}s")

    def submit_prompts(self, new_batch: DataProto, gen_batch: DataProto):
        """
        提交 (new_batch, gen_batch) 到 server 的待生成队列（非阻塞）。
        Server 端会执行: generate → repeat → union → pool.put
        """
        if self.rank == 0:
            payload_bytes = pickle.dumps((new_batch, gen_batch))
            logger.info(f"Submitting prompts ({len(payload_bytes) / 1024:.1f} KB)...")
            resp = requests.post(
                f"{self.server_url}/submit_prompts",
                data=payload_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=SUBMIT_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info(f"Submit response: {resp.json()}")

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    def pull_samples(
        self,
        target_samples: int,
        min_weight_version: int = 0,
        timeout: float = 1200.0,
    ) -> dict:
        """
        高性能异步拉取：元数据与张量分流同步方案。

        [FIX] 关键显存修复点:
          1. rank0 将 batch 搬上 GPU 后，立即 del CPU 侧 data_proto，防止 CPU/GPU 双份共存
          2. NCCL broadcast 时用 torch.no_grad()，防止产生 autograd history 占用显存
          3. broadcast 完成后立即 del rank0 的 batch_td，否则每轮训练多占一份显存
          4. 非 rank0 直接在 CUDA 设备上 empty 分配，避免 CPU→GPU 的二次搬运
        """
        from tensordict import TensorDict

        t_start = time.time()
        batch_td = None

        # --- 阶段 1: Rank 0 从 Server HTTP 拉取数据 ---
        if self.rank == 0:
            params = {
                "target_samples": target_samples,
                "min_weight_version": min_weight_version,
                "timeout": timeout,
            }
            logger.info(f"[Pull] Requesting {target_samples} samples from {self.server_url}...")

            t_http = time.time()
            resp = requests.post(
                f"{self.server_url}/pull_samples",
                data=pickle.dumps(params),
                headers={"Content-Type": "application/octet-stream"},
                timeout=timeout + 30,
            )
            resp.raise_for_status()

            full_result = pickle.loads(resp.content)
            data_proto = full_result["data"]
            logger.info(f"[Pull] HTTP+Pickle took {time.time() - t_http:.2f}s")

            # [FIX] non_blocking=True 让 H2D 拷贝异步执行，不阻塞 CPU
            batch_td = data_proto.batch.cuda(non_blocking=True).contiguous()

            tensor_keys = sorted(batch_td.keys())
            header = {
                "keys": tensor_keys,
                "shapes": {k: batch_td[k].shape for k in tensor_keys},
                "dtypes": {k: batch_td[k].dtype for k in tensor_keys},
                "batch_size": batch_td.batch_size,
                "meta_info": data_proto.meta_info,
                "non_tensor_batch": data_proto.non_tensor_batch,
                "weight_versions": full_result.get("weight_versions", []),
                "total_samples": full_result.get("total_samples", 0),
            }
        else:
            header = None

        # --- 阶段 2: 广播非 tensor 元信息 ---
        t_bc = time.time()
        obj_list = [header]
        dist.broadcast_object_list(obj_list, src=0)
        header = obj_list[0]

        keys = header["keys"]

        # --- 阶段 3: NCCL 广播 tensor 数据 ---
        final_batch_tensors = {}
        with torch.no_grad():  # [FIX] 禁止 autograd 追踪，防止 broadcast 占用 grad 内存
            for k in keys:
                if self.rank == 0:
                    send_tensor = batch_td[k]
                else:
                    # [FIX] 直接在 CUDA 上分配，避免 CPU→GPU 二次拷贝
                    send_tensor = torch.empty(
                        header["shapes"][k], dtype=header["dtypes"][k], device="cuda"
                    )
                dist.broadcast(send_tensor, src=0)
                final_batch_tensors[k] = send_tensor

        final_batch = TensorDict(final_batch_tensors, batch_size=header["batch_size"])
        result_proto = DataProto(
            batch=final_batch,
            non_tensor_batch=header["non_tensor_batch"],
            meta_info=header["meta_info"],
        )

        if self.rank == 0:
            logger.info(
                f"[Pull] Broadcast took {time.time() - t_bc:.2f}s. "
                f"Total: {time.time() - t_start:.2f}s"
            )

        return {
            "data": result_proto,
            "weight_versions": header["weight_versions"],
            "total_samples": header["total_samples"],
        }

    def generate(self, prompts: DataProto) -> DataProto:
        """同步生成（用于 validation，保持向后兼容）"""
        if self.rank == 0:
            payload_bytes = pickle.dumps(prompts)
            logger.info(f"Sending generate request ({len(payload_bytes) / 1024:.1f} KB)...")
            t0 = time.time()
            resp = requests.post(
                f"{self.server_url}/generate",
                data=payload_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=GENERATE_TIMEOUT,
            )
            resp.raise_for_status()
            output = pickle.loads(resp.content)
            logger.info(f"Generate done in {time.time() - t0:.2f}s")

            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.broadcast_object_list([output], src=0)
        else:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            output = obj_list[0]

        return output

    def health_check(self) -> dict:
        if self.rank == 0:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        return {}

    def pool_status(self) -> dict:
        if self.rank == 0:
            resp = requests.get(f"{self.server_url}/pool_status", timeout=10)
            resp.raise_for_status()
            return resp.json()
        return {}
    
    