# training_async_runtime.py
"""
TrainingAsyncRuntime - 训练侧异步运行时

核心职责：
1. 解耦同步训练循环与异步生成流水线
2. 管理训练侧与 RolloutServer 的双向异步通信
3. 本地缓存原始数据，组装 SGLang 返回结果为完整 DataProto

线程模型：
- 主线程（训练循环）：同步调用 submit_batch / get_batch
- IO 线程（独立）：运行 asyncio，处理 HTTP/WebSocket 通信
- 通过线程安全 Queue 进行跨线程通信
"""

from __future__ import annotations

import asyncio
import json
import aiohttp
import pickle
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import numpy as np
import torch
from tensordict import TensorDict

from ...protocol import DataProto
from ...utils.py_functional import timer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ...workers.config import PPOConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构定义
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CachedRequest:
    """
    训练侧本地缓存的请求元数据。
    用于在收到 SGLang 结果后，关联原始 DataProto 进行组装。
    """
    request_id: str
    new_batch: DataProto          # 原始训练数据（含 uid, ground_truth 等）
    gen_batch: DataProto          # 生成配置（含 sampling_params 等）
    weight_version: int           # 提交时的模型版本
    num_prompts: int              # prompt 数量
    n: int                        # 每个 prompt 生成 n 个 samples
    submitted_at: float           # 提交时间戳
    
    # 追踪完成状态
    received_samples: int = 0
    expected_samples: int = field(init=False)
    
    def __post_init__(self):
        self.expected_samples = self.num_prompts * self.n


@dataclass
class RawSampleResult:
    """
    从 RolloutServer 接收的原始生成结果。
    尚未组装为 DataProto，只包含 token 序列和关联信息。
    """
    request_id: str
    prompt_idx: int               # 在原始 batch 中的位置
    sample_idx: int               # 第几个 sample（0 to n-1）
    output_ids: List[int]         # 生成的 token ids
    num_input_tokens: int
    num_output_tokens: int
    finish_reason: str
    weight_version: int           # 生成时使用的模型版本


# ═══════════════════════════════════════════════════════════════════════════════
# 主类：TrainingAsyncRuntime
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingAsyncRuntime:
    """
    训练侧异步运行时：连接同步训练循环与异步生成服务。
    
    使用示例：
        runtime = TrainingAsyncRuntime(config, tokenizer, server_url)
        runtime.start()  # 启动后台 IO 线程
        
        # 预填充
        for _ in range(4):
            runtime.submit_batch(next(dataloader))
        
        # 训练循环
        for step in range(steps):
            if runtime.can_submit():
                runtime.submit_batch(next(dataloader))  # 非阻塞
            batch = runtime.get_batch(timeout=600)      # 阻塞等待
            train(batch)
        
        runtime.shutdown()
    """
    
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        server_url: str,
        # 队列容量配置
        submit_queue_size: int = 20,      # 待提交 batch 队列
        ready_queue_size: int = 2,         # 就绪 batch 队列（双缓冲）
        max_inflight_requests: int = 50,   # 最大在途 request 数
        # 组装配置
        target_samples: Optional[int] = None,  # 每批训练样本数
        staleness_tolerance: int = 2,      # 最大版本滞后容忍
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.server_url = server_url.rstrip("/")
        
        # 计算目标样本数：rollout_batch_size * rollout.n
        self.target_samples = target_samples or (
            config.data.rollout_batch_size * config.worker.rollout.n
        )
        self.staleness_tolerance = staleness_tolerance
        self.max_inflight_requests = max_inflight_requests
        
        # ═══════════════════════════════════════════════════════════════════════
        # 线程安全队列：同步侧 <-> 异步侧 通信
        # ═══════════════════════════════════════════════════════════════════════
        
        # 主线程 -> IO 线程：待提交的 batches
        self._submit_queue: Queue[DataProto] = Queue(maxsize=submit_queue_size)
        
        # IO 线程 -> 主线程：已就绪的 DataProto batches
        self._ready_queue: Queue[DataProto] = Queue(maxsize=ready_queue_size)
        
        # 主线程 -> IO 线程：weight version 更新通知
        self._weight_queue: Queue[int] = Queue()
        
        # ═══════════════════════════════════════════════════════════════════════
        # 状态管理（主线程和 IO 线程都可能访问，需加锁）
        # ═══════════════════════════════════════════════════════════════════════
        
        self._state_lock = threading.Lock()
        self._current_weight_version: int = 1
        self._inflight_requests: Dict[str, CachedRequest] = {}  # 在途请求缓存
        self._inflight_count: int = 0  # 在途 request 数（用于背压）
        
        # ═══════════════════════════════════════════════════════════════════════
        # 样本累积缓冲区（IO 线程内部使用，无需锁）
        # 收到原始 samples，累积够 target_samples 后组装为 DataProto
        # ═══════════════════════════════════════════════════════════════════════
        
        self._sample_buffer: List[DataProto] = []  # 单条 sample 的 DataProto
        self._buffered_count: int = 0  # 当前缓冲的 sample 数
        
        # ═══════════════════════════════════════════════════════════════════════
        # 线程控制
        # ═══════════════════════════════════════════════════════════════════════
        
        self._io_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        
        # 统计信息
        self._stats = {
            "submitted_batches": 0,
            "submitted_samples": 0,
            "received_samples": 0,
            "assembled_batches": 0,
            "dropped_stale_samples": 0,
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 公共 API：同步侧调用（主线程 / 训练循环）
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start(self) -> None:
        """启动后台 IO 线程。必须在训练开始前调用。"""
        if self._is_running:
            return
        
        self._is_running = True
        self._io_thread = threading.Thread(
            target=self._run_io_loop,
            name="TrainingAsyncRuntime-IO",
            daemon=True,
        )
        self._io_thread.start()
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """优雅关闭，等待未完成请求。"""
        if not self._is_running:
            return
        
        self._stop_event.set()
        self._io_thread.join(timeout=timeout)
        self._is_running = False
    
    def can_submit(self) -> bool:
        """
        检查是否可以提交新 batch。
        用于背压控制：队列满或 inflight 过多时暂停提交。
        """
        # 检查提交队列是否已满
        if self._submit_queue.full():
            return False
        
        # 检查在途请求是否超限
        with self._state_lock:
            if self._inflight_count >= self.max_inflight_requests:
                return False
        
        return True
    
    def submit_batch(self, batch: DataProto) -> bool:
        """
        提交一个 batch 到生成流水线。
        
        注意：
        - 立即返回，不阻塞（只入队，HTTP 发送在后台）
        - 如果队列满，返回 False，调用方应稍后重试
        
        Args:
            batch: 包含 prompts 的 DataProto，需含 raw_prompt_ids, uid 等
        
        Returns:
            True: 成功入队
            False: 队列满，提交失败
        """
        try:
            self._submit_queue.put_nowait(batch)
            return True
        except Full:
            return False
    
    def get_batch(self, timeout: Optional[float] = None) -> Optional[DataProto]:
        """
        阻塞等待，获取一个就绪的训练 batch。
        
        这是训练循环的核心同步点。由于有预填充和后台持续生成，
        通常能立即返回，无需等待。
        
        Args:
            timeout: 最大等待秒数，None 表示无限等待
        
        Returns:
            组装好的 DataProto，或 None（超时）
        """
        try:
            return self._ready_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def notify_weight_sync(self, new_version: int) -> None:
        """
        通知异步侧模型权重已更新。
        
        异步侧会将此版本传播到 RolloutServer，Server 据此过滤
        过期 samples（version < new_version - staleness_tolerance）。
        
        Args:
            new_version: 新的权重版本号
        """
        self._weight_queue.put_nowait(new_version)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取运行时统计信息。"""
        with self._state_lock:
            return {
                **self._stats,
                "inflight_requests": self._inflight_count,
                "current_weight_version": self._current_weight_version,
                "submit_queue_size": self._submit_queue.qsize(),
                "ready_queue_size": self._ready_queue.qsize(),
                "sample_buffer_count": self._buffered_count,
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 后台 IO 线程：异步事件循环
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _run_io_loop(self):
        """IO 线程入口：启动 asyncio 事件循环。"""
        asyncio.run(self._async_main())
    
    async def _async_main(self):
        """异步主函数：管理所有并发任务。"""
        # 创建 aiohttp session（连接池复用）
        import aiohttp
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            headers={"Content-Type": "application/json"},
        ) as session:
            
            # 并发运行多个任务
            await asyncio.gather(
                self._http_submitter_task(session),    # 持续提交请求
                self._websocket_receiver_task(),        # 持续接收结果
                self._weight_sync_task(session),        # 传播 weight 更新
                self._health_check_task(session),       # 保活检查
                return_exceptions=True,
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 任务 1：HTTP 提交器 - 持续从队列取 batch，流式发送
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _http_submitter_task(self, session: aiohttp.ClientSession):
        """
        持续消费 _submit_queue，将 DataProto 拆分为轻量请求发送。
        
        关键设计：
        - 不等待 HTTP 响应（fire-and-forget）
        - 每个 batch 生成唯一 request_id，用于后续结果关联
        - 原始 DataProto 在本地缓存，不通过网络传输
        """
        while not self._stop_event.is_set():
            # 非阻塞检查队列（避免阻塞事件循环）
            try:
                batch = self._submit_queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.001)  # 1ms 让步
                continue
            
            # 生成唯一标识
            request_id = str(uuid.uuid4())
            
            # 构造轻量提交数据（只传必要信息）
            submit_payload = self._build_submit_payload(batch, request_id)
            
            # 缓存原始数据，用于后续组装
            cached = CachedRequest(
                request_id=request_id,
                new_batch=batch,
                gen_batch=self._extract_gen_batch(batch),  # 提取生成配置
                weight_version=self._current_weight_version,
                num_prompts=len(batch),
                n=self.config.worker.rollout.n,
                submitted_at=time.time(),
            )
            
            with self._state_lock:
                self._inflight_requests[request_id] = cached
                self._inflight_count += 1
                self._stats["submitted_batches"] += 1
                self._stats["submitted_samples"] += cached.expected_samples
            
            # 异步发送，不等待响应（创建后台任务）
            asyncio.create_task(
                self._send_one_request(session, submit_payload, cached)
            )
    
    async def _send_one_request(
        self,
        session: aiohttp.ClientSession,
        payload: Dict,
        cached: CachedRequest,
    ):
        """发送单个请求，失败时重试或清理缓存。"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{self.server_url}/submit_streaming",
                    json=payload,
                ) as resp:
                    if resp.status == 202:  # Accepted
                        return  # 成功，无需处理响应
                    else:
                        # Server 错误，等待后重试
                        await asyncio.sleep(0.1 * (attempt + 1))
            except aiohttp.ClientError as e:
                # 网络错误，等待后重试
                await asyncio.sleep(0.5 * (attempt + 1))
        
        # 最终失败：清理缓存，释放资源
        with self._state_lock:
            self._inflight_requests.pop(cached.request_id, None)
            self._inflight_count -= 1
    
    def _build_submit_payload(self, batch: DataProto, request_id: str) -> Dict:
        """
        构造轻量提交数据。
        
        原则：只传 token ids 和必要配置，大 tensors（图片）不传，
        通过 request_id 在训练侧关联。
        """
        return {
            "request_id": request_id,
            "weight_version": self._current_weight_version,
            "prompts": batch.non_tensor_batch["raw_prompt_ids"].tolist(),
            "sampling_params": {
                "temperature": batch.meta_info.get("temperature", 1.0),
                "max_new_tokens": self.config.worker.rollout.response_length,
                "n": self.config.worker.rollout.n,
            },
            # 注意：不传 uid, ground_truth, image 等大/复杂数据
        }
    
    def _extract_gen_batch(self, batch: DataProto) -> DataProto:
        """从完整 batch 中提取生成所需的配置部分。"""
        # 复用现有逻辑：pop 出 gen 相关字段
        return batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 任务 2：WebSocket 接收器 - 持续接收生成结果
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _websocket_receiver_task(self):
        """
        通过 WebSocket 接收 RolloutServer 推送的完成 sample。
        
        每收到一个 sample：
        1. 解析为 RawSampleResult
        2. 关联本地缓存的原始数据
        3. 组装为单条 DataProto
        4. 累积到 buffer，够数后组装为 batch 入 ready_queue
        """
        import aiohttp
        
        # 连接 WebSocket（带自动重连）
        while not self._stop_event.is_set():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        f"{self.server_url}/ws/results",
                        heartbeat=30.0,
                    ) as ws:
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                await self._handle_server_message(data)
                            
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break  # 出错，触发重连
                            
            except Exception as e:
                # 连接失败，等待后重连
                await asyncio.sleep(5.0)
    
    async def _handle_server_message(self, data: Dict):
        """处理 Server 推送的消息。"""
        msg_type = data.get("type")
        
        if msg_type == "sample":
            # 单个 sample 完成
            raw = RawSampleResult(
                request_id=data["request_id"],
                prompt_idx=data["prompt_idx"],
                sample_idx=data["sample_idx"],
                output_ids=data["output_ids"],
                num_input_tokens=data["num_input_tokens"],
                num_output_tokens=data["num_output_tokens"],
                finish_reason=data["finish_reason"],
                weight_version=data["weight_version"],
            )
            await self._process_sample(raw)
            
        elif msg_type == "request_complete":
            # 整个 request 完成（可选：清理缓存）
            pass
    
    async def _process_sample(self, raw: RawSampleResult):
        """
        处理单个完成的 sample：关联缓存，组装 DataProto。
        """
        # 查找缓存的原始数据
        with self._state_lock:
            cached = self._inflight_requests.get(raw.request_id)
        
        if cached is None:
            # 可能已过期被清理，或 weight sync 后丢弃
            with self._state_lock:
                self._stats["dropped_stale_samples"] += 1
            return
        
        # 版本检查：过期的 sample 直接丢弃
        if raw.weight_version < self._current_weight_version - self.staleness_tolerance:
            with self._state_lock:
                self._stats["dropped_stale_samples"] += 1
            
            # 更新追踪状态
            cached.received_samples += 1
            if cached.received_samples >= cached.expected_samples:
                self._cleanup_request(raw.request_id)
            return
        
        # 组装单条 DataProto
        single_dataproto = self._assemble_single_sample(raw, cached)
        
        # 加入 buffer
        self._sample_buffer.append(single_dataproto)
        self._buffered_count += 1
        
        with self._state_lock:
            self._stats["received_samples"] += 1
        
        # 更新追踪
        cached.received_samples += 1
        
        # 检查是否凑够一个训练 batch
        if self._buffered_count >= self.target_samples:
            await self._flush_buffer()
        
        # 检查 request 是否完成
        if cached.received_samples >= cached.expected_samples:
            self._cleanup_request(raw.request_id)
    
    def _assemble_single_sample(
        self,
        raw: RawSampleResult,
        cached: CachedRequest,
    ) -> DataProto:
        """
        将单个原始结果组装为 DataProto。
        
        需要：
        1. 从 cached.new_batch 提取对应 prompt 的元信息
        2. 处理 output_ids（去除 prompt overlap）
        3. 构造 TensorDict
        4. 与原始数据 union
        """
        # 提取对应位置的原始数据
        original = cached.new_batch[raw.prompt_idx : raw.prompt_idx + 1]
        
        # 处理 output_ids：去除与 prompt 的重叠
        prompt_ids = original.non_tensor_batch["raw_prompt_ids"][0].tolist()
        output_ids = self._strip_prompt_overlap(raw.output_ids, prompt_ids)
        
        # 构造 response tensor
        response_tensor = torch.tensor(
            output_ids,
            dtype=torch.long,
            device="cpu",  # 先放 CPU，concat 时再上 GPU
        )
        
        # pad 到统一长度
        max_len = self.config.worker.rollout.response_length
        if len(response_tensor) < max_len:
            padding = torch.full((max_len - len(response_tensor),), self.tokenizer.pad_token_id, dtype=torch.long)
            response_tensor = torch.cat([response_tensor, padding])
        else:
            response_tensor = response_tensor[:max_len]
        
        # 构造单条 batch
        batch_size = 1
        response_batch = TensorDict({
            "responses": response_tensor.unsqueeze(0),  # [1, seq_len]
            "response_mask": (response_tensor != self.tokenizer.pad_token_id).unsqueeze(0),
        }, batch_size=batch_size)
        
        # repeat(interleave=True) 模拟：这里 n=1，直接返回
        # 实际应根据 raw.sample_idx 处理，简化起见假设外部已处理 n
        return original.union(DataProto(batch=response_batch))
    
    def _strip_prompt_overlap(self, output_ids: List[int], prompt_ids: List[int]) -> List[int]:
        """去除 output 中与 prompt 重叠的部分。"""
        if not output_ids or not prompt_ids:
            return output_ids
        
        max_overlap = min(len(output_ids), len(prompt_ids))
        for overlap in range(max_overlap, 0, -1):
            if prompt_ids[-overlap:] == output_ids[:overlap]:
                return output_ids[overlap:]
        return output_ids
    
    async def _flush_buffer(self):
        """
        将累积的 samples 组装为完整 batch，放入 ready_queue。
        """
        if self._buffered_count == 0:
            return
        
        # 取前 target_samples 个
        to_flush = self._sample_buffer[: self.target_samples]
        self._sample_buffer = self._sample_buffer[self.target_samples :]
        self._buffered_count -= len(to_flush)
        
        # concat 为单个 DataProto
        if len(to_flush) == 1:
            batch = to_flush[0]
        else:
            batch = DataProto.concat(to_flush)
        
        # 放入 ready_queue（阻塞直到有空位，实现背压）
        while not self._stop_event.is_set():
            try:
                self._ready_queue.put_nowait(batch)
                with self._state_lock:
                    self._stats["assembled_batches"] += 1
                return
            except Full:
                await asyncio.sleep(0.01)  # 等待主线程消费
    
    def _cleanup_request(self, request_id: str):
        """清理已完成的 request 缓存。"""
        with self._state_lock:
            if request_id in self._inflight_requests:
                del self._inflight_requests[request_id]
                self._inflight_count -= 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 任务 3：Weight Sync 传播
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _weight_sync_task(self, session: aiohttp.ClientSession):
        """监听 weight_queue，传播到 Server。"""
        import aiohttp
        
        while not self._stop_event.is_set():
            try:
                # 非阻塞检查
                new_version = self._weight_queue.get_nowait()
                
                with self._state_lock:
                    self._current_weight_version = new_version
                
                # 通知 Server
                try:
                    async with session.post(
                        f"{self.server_url}/update_min_version",
                        json={
                            "min_version": new_version - self.staleness_tolerance
                        },
                    ) as resp:
                        if resp.status != 200:
                            # 失败可重试
                            pass
                except aiohttp.ClientError:
                    pass  # 下次再试
                
            except Empty:
                await asyncio.sleep(0.1)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 任务 4：健康检查
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _health_check_task(self, session: aiohttp.ClientSession):
        """定期健康检查，可扩展为自动恢复逻辑。"""
        import aiohttp
        
        while not self._stop_event.is_set():
            try:
                async with session.get(
                    f"{self.server_url}/health",
                    timeout=5.0
                ) as resp:
                    if resp.status != 200:
                        # 记录异常
                        pass
            except Exception:
                # 记录异常
                pass
            
            await asyncio.sleep(30.0)  # 30秒一次