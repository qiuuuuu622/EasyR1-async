"""
RolloutClusterClient — 多 Server 分布式 Rollout 客户端。

业务职责:
  - 权重广播：把最新训练权重推送到集群内所有 rollout server
  - 数据拉取：并发 drain 所有 server，NCCL broadcast 给所有训练 rank
  - 容错降级：某 server 失败时把其 quota 重分配给存活 server

不再负责:
  - prompt 提交的调度决策（已移至 ClusterDispatcher，由 PromptProducer 持有）

实现约束（FSDP/NCCL 决定，非业务逻辑）:
  - FSDP gather（get_model_state_dict）需要所有 rank 参与，因为模型权重分片在各 rank 上
  - HTTP I/O 只由 rank 0 执行，其余 rank 在 barrier 等待，以保持训练侧步调一致
  - pull 到的 tensor 通过 NCCL broadcast 分发给所有训练 rank

对外接口与 RolloutClient 完全一致：
  push_weights / pull_samples / generate / health_check / pool_status
"""

from __future__ import annotations

import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ....protocol import DataProto
from .cluster_topology import ClusterTopology
from .server_endpoint import ServerEndpoint, NetworkError

logger = logging.getLogger(__name__)

_DEFAULT_PULL_TIMEOUT = 1200.0


class RolloutClusterClient:
    """
    drop-in replacement for RolloutClient，扇出到 N 个 RolloutServer。

    submit_prompts 已从此类移除，由 ClusterDispatcher（PromptProducer 持有）负责。
    """

    def __init__(
        self,
        server_urls: list[str],
        fsdp_module: FSDP,
        rank: int = 0,
        health_check_interval: float = 10.0,
    ):
        self.fsdp_module    = fsdp_module
        self.rank           = rank
        self.weight_version = 0

        self._topology = ClusterTopology(
            server_urls=server_urls,
            health_check_interval=health_check_interval,
        )
        self._fanout_pool = ThreadPoolExecutor(
            max_workers=len(server_urls),
            thread_name_prefix="cluster-fanout",
        )

        if self.rank == 0:
            self._topology.wait_until_ready()
            self._topology.start_health_watcher()

        logger.info(
            f"[ClusterClient] Ready: {len(server_urls)} servers, rank={rank}"
        )

    # ── push_weights ──────────────────────────────────────────────────────────

    def push_weights(
        self,
        use_param_offload: bool = False,
        flush_cache: bool = False,
    ) -> None:
        """
        所有 rank 参与 FSDP gather；rank 0 并行广播给所有健康 server；
        最后全员 barrier。

        返回推送后的 weight_version（调用方用于 notify_weight_version）。
        """
        from ....utils.fsdp_utils import load_fsdp_model, offload_fsdp_model

        if use_param_offload:
            load_fsdp_model(self.fsdp_module)

        t0 = time.time()
        logger.info("[ClusterClient] Gathering FSDP state dict …")
        state_dict = get_model_state_dict(self.fsdp_module)

        # 逐 tensor 转 CPU，避免 GPU 上同时持有两份完整权重
        cpu_sd: dict = {}
        for name, tensor in state_dict.items():
            cpu_sd[name] = (
                tensor.full_tensor().cpu()
                if hasattr(tensor, "full_tensor")
                else tensor.cpu()
            )
        del state_dict

        if use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self.rank == 0:
            self.weight_version += 1
            payload_bytes = pickle.dumps({
                "weight_version": self.weight_version,
                "state_dict":     cpu_sd,
                "flush_cache":    flush_cache,
            })
            del cpu_sd
            self._fanout_push(payload_bytes)
            del payload_bytes
        else:
            del cpu_sd

        if dist.is_initialized():
            dist.barrier()

        logger.info(
            f"[ClusterClient] push_weights done in {time.time() - t0:.2f}s "
            f"(wv={self.weight_version})"
        )
        return self.weight_version

    def _fanout_push(self, payload_bytes: bytes) -> None:
        """rank 0 内部：并行 push 到所有健康 server，网络失败的标记 is_healthy=False。"""
        healthy = self._topology.healthy_endpoints()
        size_mb = len(payload_bytes) / 1024 / 1024
        logger.info(
            f"[ClusterClient] Pushing weights v{self.weight_version} "
            f"({size_mb:.1f} MB) to {len(healthy)} servers …"
        )
        futures = {
            self._fanout_pool.submit(ep.push_weights, payload_bytes): ep
            for ep in healthy
        }
        for fut in as_completed(futures):
            ep = futures[fut]
            try:
                fut.result()
                ep.weight_version = self.weight_version
                logger.info(f"[ClusterClient] {ep.url} → wv={self.weight_version}")
            except NetworkError as e:
                ep.is_healthy = False
                logger.warning(f"[ClusterClient] push_weights to {ep.url} failed: {e}")

    # ── pull_samples ──────────────────────────────────────────────────────────

    def pull_samples(
        self,
        target_samples: int,
        min_weight_version: int = 0,
        timeout: float = _DEFAULT_PULL_TIMEOUT,
    ) -> dict:
        """
        rank 0 并行从所有健康 server 拉取，合并后 NCCL broadcast 给所有 rank。

        返回 {"data": DataProto, "weight_versions": list[int], "total_samples": int}
        """
        t_start = time.time()
        batch_td: Optional[TensorDict] = None
        header:   Optional[dict]       = None

        # ── Stage 1: rank 0 拉取并搬到 GPU ────────────────────────────────────
        if self.rank == 0:
            merged       = self._parallel_pull(target_samples, min_weight_version, timeout)
            data_proto   = merged["data"]

            batch_td     = data_proto.batch.cuda(non_blocking=True).contiguous()
            tensor_keys  = sorted(batch_td.keys())
            header = {
                "keys":             tensor_keys,
                "shapes":           {k: batch_td[k].shape for k in tensor_keys},
                "dtypes":           {k: batch_td[k].dtype for k in tensor_keys},
                "batch_size":       batch_td.batch_size,
                "meta_info":        data_proto.meta_info,
                "non_tensor_batch": data_proto.non_tensor_batch,
                "weight_versions":  merged["weight_versions"],
                "total_samples":    merged["total_samples"],
            }
            del data_proto, merged

        # ── Stage 2: broadcast 非 tensor 元信息 ───────────────────────────────
        obj_list = [header]
        dist.broadcast_object_list(obj_list, src=0)
        header = obj_list[0]

        # ── Stage 3: NCCL broadcast tensors ───────────────────────────────────
        final_tensors: dict = {}
        with torch.no_grad():
            for k in header["keys"]:
                if self.rank == 0:
                    t = batch_td[k]
                else:
                    t = torch.empty(
                        header["shapes"][k],
                        dtype=header["dtypes"][k],
                        device="cuda",
                    )
                dist.broadcast(t, src=0)
                final_tensors[k] = t

        if self.rank == 0:
            del batch_td

        result_proto = DataProto(
            batch=TensorDict(final_tensors, batch_size=header["batch_size"]),
            non_tensor_batch=header["non_tensor_batch"],
            meta_info=header["meta_info"],
        )

        if self.rank == 0:
            logger.info(
                f"[ClusterClient] pull_samples done in {time.time() - t_start:.2f}s, "
                f"samples={header['total_samples']}, wvs={header['weight_versions']}"
            )

        return {
            "data":            result_proto,
            "weight_versions": header["weight_versions"],
            "total_samples":   header["total_samples"],
        }

    def _parallel_pull(
        self,
        target_samples: int,
        min_weight_version: int,
        timeout: float,
    ) -> dict:
        """
        全局聚合拉取：向所有健康 server 并发发 drain 请求，
        把各 server 当前池内有的样本汇总，不够就轮询直到凑够 target_samples。
        """
        all_data: list[DataProto] = []
        all_wvs:  list[int]       = []
        total         = 0
        deadline      = time.time() + timeout
        poll_interval = 1.0

        while total < target_samples:
            if time.time() > deadline:
                raise RuntimeError(
                    f"[ClusterClient] pull_samples timeout after {timeout:.0f}s "
                    f"(collected {total}/{target_samples})"
                )

            healthy = self._topology.healthy_endpoints()
            if not healthy:
                raise RuntimeError("[ClusterClient] No healthy servers during pull.")

            futures = {
                self._fanout_pool.submit(
                    self._drain_one, ep, target_samples - total, min_weight_version, timeout
                ): ep
                for ep in healthy
            }

            got_anything = False
            for fut in as_completed(futures):
                ep     = futures[fut]
                result = fut.result()
                if result is None or result["total_samples"] == 0:
                    continue
                all_data.append(result["data"])
                all_wvs.extend(result.get("weight_versions", []))
                total += result["total_samples"]
                got_anything = True
                logger.info(
                    f"[ClusterClient] drain {ep.url}: +{result['total_samples']} "
                    f"(total {total}/{target_samples})"
                )
                if total >= target_samples:
                    break

            if not got_anything:
                elapsed = time.time() - (deadline - timeout)
                if int(elapsed) % 30 < poll_interval:
                    logger.info(
                        f"[ClusterClient] waiting for samples "
                        f"({total}/{target_samples}) "
                        f"elapsed={elapsed:.0f}s"
                    )
                time.sleep(poll_interval)

        merged = DataProto.concat(all_data) if len(all_data) > 1 else all_data[0]
        if len(merged) > target_samples:
            merged = merged[:target_samples]
            total  = target_samples

        logger.info(f"[ClusterClient] pull merged: {total} samples, wvs={all_wvs}")
        return {"data": merged, "weight_versions": all_wvs, "total_samples": total}

    @staticmethod
    def _drain_one(
        ep: ServerEndpoint,
        max_samples: int,
        min_weight_version: int,
        timeout: float,
    ) -> Optional[dict]:
        """
        向单个 server 发 drain 请求：有多少出多少，不阻塞等待。
        失败返回 None 并标记 is_healthy=False。
        """
        params = {
            "target_samples":     max_samples,
            "min_weight_version": min_weight_version,
            "timeout":            timeout,
            "drain":              True,
        }
        try:
            return ep.pull_samples(params, server_timeout=timeout)
        except TimeoutError:
            # drain 模式下 408 表示当前池子为空，不是真正的错误
            return None
        except (NetworkError, Exception) as exc:
            logger.warning(f"[ClusterClient] drain {ep.url} failed: {exc}")
            ep.is_healthy = False
            return None

    # ── generate (validation) ─────────────────────────────────────────────────

    def generate(self, prompts: DataProto) -> DataProto:
        """同步推理，发给第一个健康 server，结果 broadcast 给所有 rank。"""
        if self.rank == 0:
            ep = self._topology.healthy_endpoints()[0]
            payload_bytes = pickle.dumps(prompts)
            t0 = time.time()
            logger.info(f"[ClusterClient] generate → {ep.url} …")
            output = ep.generate(payload_bytes)
            logger.info(f"[ClusterClient] generate done in {time.time() - t0:.2f}s")

            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.broadcast_object_list([output], src=0)
        else:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            output = obj_list[0]

        return output

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        if self.rank != 0:
            return {}
        return {
            "weight_version": self.weight_version,
            **self._topology.status(),
        }

    def pool_status(self) -> dict:
        if self.rank != 0:
            return {}
        healthy = self._topology.healthy_endpoints()
        futures = {
            self._fanout_pool.submit(ep.pool_status): ep
            for ep in healthy
        }
        results = []
        for fut in as_completed(futures):
            ep = futures[fut]
            results.append({"url": ep.url, **fut.result()})
        return {"servers": results}

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._topology.shutdown()
        self._fanout_pool.shutdown(wait=False)
        logger.info("[ClusterClient] Shutdown complete.")