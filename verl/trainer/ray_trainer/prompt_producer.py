"""
PromptProducer — 完全自驱的 prompt 提交后台线程。

职责（仅此而已）:
  - 从 StatefulDataLoader 持续取 batch，构造 (new_batch, gen_batch)
  - 通过 ClusterDispatcher 提交到 rollout cluster
  - 遇到 ServerBusyError 自己退避重试，不需要 Trainer 感知
  - 在 meta_info 中携带当前 weight_version（由 Trainer notify）

设计决策：
  - 背压完全由 ServerBusyError（即 503）驱动，不维护 pending 计数
  - 不感知 HTTP，不感知 server 数量，不感知训练步数
  - Trainer 唯一入口：notify_weight_version(v)
  - Checkpoint 接口：state_dict() / load_state_dict()，Trainer 在 save/load 时调用

外部契约：
    Trainer ──notify_weight_version(v)──▶ Producer（记录，下次 submit 携带）
    Producer ──submit(batch, wv)────────▶ ClusterDispatcher
    ClusterDispatcher ──ServerBusyError─▶ Producer（退避 backoff 秒后重试）
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Optional, TYPE_CHECKING

import numpy as np
from torchdata.stateful_dataloader import StatefulDataLoader

from ...protocol import DataProto
from ...workers.rollout.cluster.cluster_dispatcher import ClusterDispatcher
from ...workers.rollout.cluster.server_endpoint import ServerBusyError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from ..config import PPOConfig

logger = logging.getLogger(__name__)


class PromptProducer:
    """
    后台线程，持续把 prompt 送进 rollout cluster。

    用法：
        producer = PromptProducer(dataloader, dispatcher, config, tokenizer)
        producer.start()

        # Trainer 每次 push_weights 后：
        producer.notify_weight_version(new_version)

        # Trainer 结束时：
        producer.stop()

        # Checkpoint：
        torch.save(producer.state_dict(), path)
        producer.load_state_dict(torch.load(path))
    """

    def __init__(
        self,
        dataloader: StatefulDataLoader,
        dispatcher: ClusterDispatcher,
        config: "PPOConfig",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"] = None,
        backoff: float = 2.0,
    ):
        self._dataloader  = dataloader
        self._dispatcher  = dispatcher
        self._config      = config
        self._tokenizer   = tokenizer
        self._processor   = processor
        self._backoff     = backoff

        # weight_version 由 Trainer 通过 notify_weight_version() 写入
        # _version_lock 保证跨线程可见性
        self._current_weight_version: int = 1
        self._version_lock = threading.Lock()

        # 生命周期控制
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # 从 dataloader 建立 iterator，checkpoint 恢复时重建
        self._data_iter = iter(self._dataloader)

        logger.info("[PromptProducer] Initialized.")

    # ── Trainer 调用的唯一入口 ────────────────────────────────────────────────

    def notify_weight_version(self, version: int) -> None:
        """
        Trainer 在 push_weights() 完成后立即调用。
        Producer 记录 version，下一次 submit 时携带在 meta_info 里。
        非阻塞，线程安全，立即返回。
        """
        with self._version_lock:
            self._current_weight_version = version
        logger.debug(f"[PromptProducer] weight_version updated to {version}.")

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """启动后台提交线程。"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[PromptProducer] Already running, ignoring start().")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="prompt-producer",
        )
        self._thread.start()
        logger.info("[PromptProducer] Background thread started.")

    def stop(self, timeout: float = 10.0) -> None:
        """
        通知后台线程停止，等待其退出。
        当前正在进行的 submit 会完成后才退出。
        """
        logger.info("[PromptProducer] Stopping...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("[PromptProducer] Thread did not exit within timeout.")
        logger.info("[PromptProducer] Stopped.")

    # ── Checkpoint 接口 ───────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """
        返回可序列化的状态，供 Trainer 在 _save_checkpoint() 时保存。
        包含 dataloader 位置，恢复后可从断点继续取数据。
        """
        return {
            "dataloader": self._dataloader.state_dict(),
            "weight_version": self._current_weight_version,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        恢复状态，供 Trainer 在 _load_checkpoint() 时调用。
        必须在 start() 之前调用。
        """
        if "dataloader" in state:
            self._dataloader.load_state_dict(state["dataloader"])
            self._data_iter = iter(self._dataloader)
            logger.info("[PromptProducer] Dataloader state restored.")
        if "weight_version" in state:
            with self._version_lock:
                self._current_weight_version = state["weight_version"]

    # ── 后台主循环 ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """
        后台线程主循环。完全自驱：
          - 正常：取 batch → submit → 继续
          - ServerBusyError：退避 backoff 秒后重试同一 batch
          - 其他异常：记录日志，退避后跳过（丢弃当前 batch，取下一个）
        """
        logger.info("[PromptProducer] Loop started.")
        pending_batch: Optional[tuple] = None   # (new_batch, gen_batch)，失败时保留重试

        while not self._stop_event.is_set():
            try:
                # 没有待重试的 batch 时，从 dataloader 取新的
                if pending_batch is None:
                    pending_batch = self._build_batch()

                new_batch, gen_batch = pending_batch

                with self._version_lock:
                    wv = self._current_weight_version

                self._dispatcher.submit(new_batch, gen_batch, weight_version=wv)
                pending_batch = None   # 提交成功，清空

                logger.debug(
                    f"[PromptProducer] Submitted {len(new_batch)} prompts "
                    f"(wv={wv})."
                )

            except ServerBusyError:
                # 背压：所有 server 当前都满，退避后用同一个 batch 重试
                logger.debug(
                    f"[PromptProducer] All servers busy, "
                    f"backing off {self._backoff}s."
                )
                self._stop_event.wait(timeout=self._backoff)

            except RuntimeError as e:
                # 所有 server 都 unhealthy，严重错误，退避后继续尝试
                # （不退出循环，等待 ClusterTopology 把 server 恢复健康）
                logger.error(
                    f"[PromptProducer] No healthy servers: {e}. "
                    f"Retrying in {self._backoff}s."
                )
                self._stop_event.wait(timeout=self._backoff)

            except StopIteration:
                # dataloader 耗尽（理论上 _build_batch 内部会自动重置，保险起见）
                logger.info("[PromptProducer] Dataloader exhausted, resetting.")
                self._data_iter = iter(self._dataloader)
                pending_batch = None

            except Exception as e:
                # 未预期的错误，记录并跳过当前 batch
                logger.error(
                    f"[PromptProducer] Unexpected error: {e}. "
                    f"Skipping batch, retrying in {self._backoff}s.",
                    exc_info=True,
                )
                pending_batch = None
                self._stop_event.wait(timeout=self._backoff)

        logger.info("[PromptProducer] Loop exited.")

    # ── batch 构造（从 AsyncPromptPipeline._submit_one_batch 迁移）────────────

    def _build_batch(self) -> tuple:
        """
        从 dataloader 取一个 raw batch，构造 (new_batch, gen_batch)。

        new_batch: 包含 uid、ground_truth 等训练侧字段
        gen_batch: 包含 input_ids、sampling_params 等推理侧字段
        """
        cfg = self._config

        # 取数据，epoch 结束自动重置
        try:
            batch_dict = next(self._data_iter)
        except StopIteration:
            logger.info("[PromptProducer] Epoch end, resetting dataloader.")
            self._data_iter = iter(self._dataloader)
            batch_dict = next(self._data_iter)

        meta_info = {
            "min_pixels": cfg.data.min_pixels,
            "max_pixels": cfg.data.max_pixels,
            "video_fps":  cfg.data.video_fps,
        }
        new_batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)

        # 为每条样本生成唯一 id，用于 online filtering 和 staleness 追踪
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
            dtype=object,
        )

        # 分离推理侧字段，gen_batch 只送给 rollout server
        gen_batch = new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        gen_batch.meta_info["eos_token_id"] = self._tokenizer.eos_token_id
        gen_batch.meta_info["pad_token_id"] = self._tokenizer.pad_token_id

        return new_batch, gen_batch