"""
RolloutServer — 只做 HTTP 层

职责边界：
  - 持有并组装三个核心组件：DispatchQueue、ExperiencePool、PromptScheduler
  - 提供 FastAPI 路由，每条路由只做：反序列化 → 调用组件方法 → 序列化返回
  - 管理 lifespan：启动/停止 PromptScheduler asyncio Task
  - 持有 weight_version 和 weight_updating 状态，供路由和调度器共同读写
  - 不含任何业务逻辑（推理、队列管理、样本合并均在各自组件中）

依赖关系（单向）：
  RolloutServer
    ├── DispatchQueue   (dispatch_queue.py)
    ├── ExperiencePool  (experience_pool.py)
    └── PromptScheduler (prompt_scheduler.py)
                └── SGLangRollout (sglang_rollout.py)
"""

import asyncio
import logging
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from .dispatch_queue import DispatchQueue, QueueFullError
from .experience_pool import ExperiencePool
from .batch_scheduler import BatchScheduler
from .rollout_latency_tracker import NoOpLatencyTracker, PromptLatencyTracker
from .sglang_rollout import SGLangRollout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# HTTP handler 线程池：用于在 asyncio event loop 中卸载阻塞的序列化/反序列化操作
_http_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="http-exec")


def _deserialize(body: bytes):
    return pickle.loads(body)


def _serialize(obj) -> bytes:
    return pickle.dumps(obj)


class _AccessLogFilter(logging.Filter):
    """
    过滤掉无业务价值的高频 access log：
      - /health        任意状态码（心跳轮询）
      - /pool_status   任意状态码（状态轮询）
      - /submit_prompts 503（队列背压，PromptProducer 会自动退避重试）
    """
    _SUPPRESS = {
        ("/health",      None),
        ("/pool_status", None),
        ("/submit_prompts", 503),
    }

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for path, code in self._SUPPRESS:
            if path in msg and (code is None or f" {code} " in msg):
                return False
        return True
    

class RolloutServer:
    """
    EasyR1 Async Rollout Server。

    启动流程：
      1. __init__：初始化 SGLangRollout、DispatchQueue、ExperiencePool、PromptScheduler
      2. run()：启动 uvicorn，lifespan startup 中创建调度 Task
      3. 外部调用 /update_weights → 推送权重 → 解锁调度
      4. 外部调用 /submit_prompts → 入 DispatchQueue
      5. 调度器持续消费 DispatchQueue → 推理 → 写 ExperiencePool
      6. 外部调用 /pull_samples → 从 ExperiencePool 阻塞取样本

    参数说明：
      max_concurrent_prompts: SGLang 同时处理的最大 prompt 数，
        根据显存和 response_length 调整，控制 KV cache 峰值。
    """

    def __init__(
        self,
        model_path: str,
        rollout_config,
        tokenizer,
        processor=None,
        host: str = "0.0.0.0",
        port: int = 8000,
        pool_max_samples: int = 5000,
        max_pending_prompts: int = 3000,
        enable_tracking: bool = False,
    ):
        self.host = host
        self.port = port
        self.weight_version: int = -1

        # 供 PromptScheduler 感知权重更新暂停信号（threading.Event，跨线程可见）
        self._weight_updating = threading.Event()

        # ── 推理引擎 ────────────────────────────────────────────────────
        logger.info(f"Initializing SGLangRollout (model={model_path})")
        self._rollout = SGLangRollout(
            model_path=model_path,
            config=rollout_config,
            tokenizer=tokenizer,
            processor=processor,
            server_mode=True,
        )

        # ── 三组件 ──────────────────────────────────────────────────────
        self._dispatch_queue = DispatchQueue(
            max_pending_prompts=max_pending_prompts,
            default_n=rollout_config.n
        )
        self._experience_pool = ExperiencePool(
            max_ready_samples=pool_max_samples,
        )
        latency_tracker = (
            PromptLatencyTracker(log_every_n=256)
            if enable_tracking
            else NoOpLatencyTracker()
        )
        self._scheduler = BatchScheduler(
            rollout=self._rollout,
            dispatch_queue=self._dispatch_queue,
            experience_pool=self._experience_pool,
            rollout_n=rollout_config.n,
            latency_tracker=latency_tracker,
            max_batch_prompts=16,         
            max_batch_cost=32768.0,       
            max_inflight_cost=131072.0,   
            batch_wait_timeout_s=0.01,
        )

        # ── FastAPI ─────────────────────────────────────────────────────
        self.app = FastAPI(title="EasyR1 Async Rollout Server")
        self._register_routes()

        # asyncio Task 句柄，在 lifespan 中赋值
        self._scheduler_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # 路由注册
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self.app

        # ── lifespan ────────────────────────────────────────────────────
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # startup
            logger.info("[Startup] creating PromptScheduler task...")
            self._scheduler_task = asyncio.create_task(
                self._scheduler.run(can_schedule_fn=self._can_schedule)
            )
            logger.info("[Startup] PromptScheduler task started.")
            yield
            # shutdown
            logger.info("[Shutdown] stopping PromptScheduler...")
            self._scheduler.stop()
            self._dispatch_queue.clear()
            self._experience_pool.clear()
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            logger.info("[Shutdown] done.")

        app.router.lifespan_context = lifespan

        # ── /health ─────────────────────────────────────────────────────
        @app.get("/health")
        def health():
            return {
                "status": "ok",
                "weight_version": self.weight_version,
                "weight_updating": self._weight_updating.is_set(),
                "scheduler_alive": (
                    self._scheduler_task is not None
                    and not self._scheduler_task.done()
                ),
                **self._scheduler.stats(),
                **self._dispatch_queue.stats(),
                **self._experience_pool.stats(),
            }

        # ── /weight_version ─────────────────────────────────────────────
        @app.get("/weight_version")
        def weight_version():
            return {"weight_version": self.weight_version}

        # ── /pool_status ────────────────────────────────────────────────
        @app.get("/pool_status")
        def pool_status():
            return {
                "weight_version": self.weight_version,
                **self._dispatch_queue.stats(),
                **self._experience_pool.stats(),
            }

        # ── /latency_stats ──────────────────────────────────────────────
        @app.get("/latency_stats")
        def latency_stats():
            return self._scheduler.latency_tracker.stats()

        # ── /debug_batch ────────────────────────────────────────────────
        @app.get("/debug_batch")
        def debug_batch():
            """诊断接口：查看 ExperiencePool 队头 chunk 的张量形状。"""
            with self._experience_pool._lock:
                if not self._experience_pool._chunks:
                    return {"error": "no ready chunks"}

                chunk = self._experience_pool._chunks[0]
                data = chunk.data

                batch_keys = list(data.batch.keys()) if data.batch is not None else []
                shapes = {k: list(data.batch[k].shape) for k in batch_keys}
                old_log_probs_shape = (
                    list(data.batch["old_log_probs"].shape)
                    if "old_log_probs" in batch_keys
                    else "NOT FOUND"
                )

                return {
                    "num_ready_chunks": len(self._experience_pool._chunks),
                    "chunk_weight_version": chunk.weight_version,
                    "chunk_num_samples": chunk.num_samples,
                    "batch_keys": batch_keys,
                    "batch_shapes": shapes,
                    "old_log_probs_shape": old_log_probs_shape,
                }

        # ── /update_weights ─────────────────────────────────────────────
        @app.post("/update_weights")
        async def update_weights(request: Request):
            """
            接收训练侧推送的新权重，加载后更新 weight_version。

            流程：
              1. 反序列化 payload（在线程池中执行，不阻塞 event loop）
              2. 版本校验：新版本 <= 当前版本则跳过
              3. 设置 _weight_updating，暂停调度器分发新 prompt
              4. 调用 SGLang 内部接口加载权重
              5. 清除 _weight_updating，恢复调度
            """
            body = await request.body()
            loop = asyncio.get_event_loop()
            try:
                payload = await loop.run_in_executor(_http_executor, _deserialize, body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Deserialize failed: {e}")
            del body

            new_version = payload["weight_version"]
            if new_version <= self.weight_version:
                logger.info(
                    f"[update_weights] skip: new={new_version} <= current={self.weight_version}"
                )
                return {"status": "skipped", "weight_version": self.weight_version}

            state_dict = payload["state_dict"]
            flush_cache = payload.get("flush_cache", True)
            del payload

            logger.info(
                f"[update_weights] loading version={new_version} "
                f"({len(state_dict)} tensors)..."
            )

            self._weight_updating.set()
            t0 = time.time()
            try:
                await self._load_weights_sglang(state_dict, flush_cache=flush_cache)
                self.weight_version = new_version
                logger.info(f"[update_weights] weight_version={self.weight_version}")
            except Exception as e:
                logger.error(f"[update_weights] FAILED: {e}", exc_info=True)
                raise
            finally:
                self._weight_updating.clear()

            del state_dict
            elapsed = time.time() - t0
            logger.info(f"[update_weights] done in {elapsed:.2f}s")
            return {
                "status": "ok",
                "weight_version": self.weight_version,
                "elapsed_seconds": elapsed,
            }

        # ── /submit_prompts ─────────────────────────────────────────────
        @app.post("/submit_prompts")
        async def submit_prompts(request: Request):
            """
            接收训练侧提交的 (new_batch, gen_batch)，入 DispatchQueue（非阻塞）。

            HTTP handler 立即返回 request_id，推理在后台异步进行。
            """
            body = await request.body()
            loop = asyncio.get_event_loop()
            try:
                payload = await loop.run_in_executor(_http_executor, _deserialize, body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Deserialize failed: {e}")
            del body

            if self.weight_version < 0:
                raise HTTPException(status_code=503, detail="No weights loaded yet.")

            new_batch, gen_batch = payload
            del payload

            try:
                request_id = await loop.run_in_executor(
                    _http_executor,
                    self._dispatch_queue.submit,
                    new_batch,
                    gen_batch,
                    self.weight_version,
                )
            except QueueFullError as e:
                # 队列满 → 503，PromptProducer 收到 ServerBusyError 后退避重试
                # 不标记 server unhealthy，server 本身是正常的
                raise HTTPException(status_code=503, detail=str(e))

            return {
                "status": "queued",
                "request_id": request_id,
                "weight_version": self.weight_version,
                **self._dispatch_queue.stats(),
                **self._experience_pool.stats(),
            }

        # ── /pull_samples ───────────────────────────────────────────────
        @app.post("/pull_samples")
        async def pull_samples(request: Request):
            """
            从 ExperiencePool 取样本。

            参数（pickle'd dict）：
              target_samples    : 期望的样本数（默认 100）
              min_weight_version: 过滤过期 chunk 的版本下限（默认 0）
              timeout           : 等待超时秒数（默认 1200）
              drain             : True  = 有多少出多少立即返回，池空返回 408
                                          （多 server 场景，RolloutClusterClient 做全局聚合）
                                  False = 阻塞等到凑够 target_samples（默认，单 server 场景）
            """
            body = await request.body()
            loop = asyncio.get_event_loop()
            try:
                params = await loop.run_in_executor(_http_executor, _deserialize, body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Deserialize failed: {e}")
            del body

            target  = params.get("target_samples", 100)
            min_wv  = params.get("min_weight_version", 0)
            timeout = params.get("timeout", 1200.0)
            drain   = params.get("drain", False)

            result = await loop.run_in_executor(
                _http_executor,
                self._experience_pool.pull,
                target,
                min_wv,
                timeout,
                drain,
            )
            if result is None:
                raise HTTPException(status_code=408, detail="Pull timeout.")

            response_bytes = await loop.run_in_executor(
                _http_executor, _serialize, result
            )
            return Response(
                content=response_bytes,
                media_type="application/octet-stream",
            )

        # ── /generate ───────────────────────────────────────────────────
        @app.post("/generate")
        async def generate(request: Request):
            """
            同步生成接口，用于 validation（不经过队列，直接推理后返回）。
            """
            body = await request.body()
            loop = asyncio.get_event_loop()
            try:
                prompts = await loop.run_in_executor(_http_executor, _deserialize, body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Deserialize failed: {e}")
            del body

            if self.weight_version < 0:
                raise HTTPException(status_code=503, detail="No weights loaded yet.")

            output = await self._rollout.generate_sequences(prompts=prompts)
            response_bytes = await loop.run_in_executor(
                _http_executor, _serialize, output
            )
            return Response(
                content=response_bytes,
                media_type="application/octet-stream",
            )

    # ------------------------------------------------------------------
    # 调度门控
    # ------------------------------------------------------------------

    def _can_schedule(self) -> bool:
        """
        PromptScheduler 唯一的调度开关。

        封装所有门控条件，调度器只问"能不能调度"，不感知原因：
          - weight_version >= 0 : 权重已完成首次加载
          - not weight_updating  : 当前没有权重热更新在进行
        如需增加新条件（如显存水位、限流…），只改这里。
        """
        return (
            self.weight_version >= 0
            and not self._weight_updating.is_set()
        )

    # ------------------------------------------------------------------
    # 权重加载（SGLang 内部接口）
    # ------------------------------------------------------------------

    async def _load_weights_sglang(
        self,
        state_dict: dict,
        flush_cache: bool = True,
    ) -> None:
        """通过 SGLang Engine 内部接口热更新权重。"""
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
        from sglang.srt.utils import MultiprocessingSerializer

        engine = self._rollout.inference_engine
        tp_size = engine.server_args.tp_size

        logger.info(f"[_load_weights] {len(state_dict)} tensors, tp={tp_size}")

        named_tensors = [
            (name, tensor)
            for name, tensor in state_dict.items()
        ]

        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(tp_size)
            ],
            load_format=None,
            flush_cache=flush_cache,
        )

        await engine.tokenizer_manager.update_weights_from_tensor(obj, None)

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def run(self) -> None:
        logging.getLogger("uvicorn.access").addFilter(_AccessLogFilter())
        logger.info(f"Starting RolloutServer on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )