import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
import logging
logger = logging.getLogger(__name__)


@dataclass
class PromptTiming:
    prompt_id: str
    request_id: str
    weight_version: int
    submit_ts: float = 0.0
    slot_enter_ts: float = 0.0
    gen_start_ts: float = 0.0
    gen_end_ts: float = 0.0
    complete_ts: float = 0.0

    @property
    def queue_wait(self) -> float:
        return self.slot_enter_ts - self.submit_ts if self.slot_enter_ts else float("nan")

    @property
    def gen_time(self) -> float:
        return self.gen_end_ts - self.gen_start_ts if self.gen_end_ts else float("nan")

    @property
    def merge_time(self) -> float:
        return self.complete_ts - self.gen_end_ts if self.complete_ts else float("nan")

    @property
    def e2e_time(self) -> float:
        return self.complete_ts - self.submit_ts if self.complete_ts else float("nan")


class NoOpLatencyTracker:
    """Drop-in replacement when latency tracking is disabled."""
    def on_submit(self, *args, **kwargs): pass
    def on_slot_enter(self, *args, **kwargs): pass
    def on_gen_start(self, *args, **kwargs): pass
    def on_gen_end(self, *args, **kwargs): pass
    def on_complete(self, *args, **kwargs): pass
    def stats(self): return {}

class PromptLatencyTracker:

    def __init__(
        self,
        window: int = 2000,
        log_every_n: int = 200,
    ):
        self._lock = threading.Lock()
        self._inflight: dict[str, PromptTiming] = {}

        self._queue_wait_buf: deque[float] = deque(maxlen=window)
        self._gen_time_buf: deque[float] = deque(maxlen=window)
        self._merge_time_buf: deque[float] = deque(maxlen=window)
        self._e2e_time_buf: deque[float] = deque(maxlen=window)

        self._total_completed = 0
        self._window = window
        self._log_every_n = log_every_n
        self._last_log_count = 0
        self._last_log_ts = time.perf_counter()

    # ------------------------------------------------------------------ #
    #  埋点 API                                                           #
    # ------------------------------------------------------------------ #

    def on_submit(self, prompt_id: str, request_id: str, weight_version: int) -> PromptTiming:
        pt = PromptTiming(
            prompt_id=prompt_id,
            request_id=request_id,
            weight_version=weight_version,
            submit_ts=time.perf_counter(),
        )
        with self._lock:
            self._inflight[prompt_id] = pt
        return pt

    def on_slot_enter(self, prompt_id: str) -> None:
        with self._lock:
            pt = self._inflight.get(prompt_id)
            if pt:
                pt.slot_enter_ts = time.perf_counter()

    def on_gen_start(self, prompt_id: str) -> None:
        with self._lock:
            pt = self._inflight.get(prompt_id)
            if pt:
                pt.gen_start_ts = time.perf_counter()

    def on_gen_end(self, prompt_id: str) -> None:
        with self._lock:
            pt = self._inflight.get(prompt_id)
            if pt:
                pt.gen_end_ts = time.perf_counter()

    def on_complete(self, prompt_id: str) -> Optional[PromptTiming]:
        with self._lock:
            pt = self._inflight.pop(prompt_id, None)
            if pt is None:
                return None
            pt.complete_ts = time.perf_counter()

            def _push(buf, val):
                if not math.isnan(val) and val >= 0:
                    buf.append(val)

            _push(self._queue_wait_buf, pt.queue_wait)
            _push(self._gen_time_buf,   pt.gen_time)
            _push(self._merge_time_buf, pt.merge_time)
            _push(self._e2e_time_buf,   pt.e2e_time)
            self._total_completed += 1

            since_count = self._total_completed - self._last_log_count
            should_log = since_count >= self._log_every_n
            if should_log:
                self._last_log_count = self._total_completed
                self._last_log_ts = time.perf_counter()
                snapshot = self._build_summary_locked(since_count)
            else:
                snapshot = None

        if snapshot:
            self._log_summary(snapshot)

        return pt

    # ------------------------------------------------------------------ #
    #  统计查询                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _percentiles(buf: deque) -> dict:
        if not buf:
            return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None,
                    "min": None, "max": None, "std": None}
        data = sorted(buf)
        n = len(data)

        def pct(p):
            idx = (p / 100) * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            return data[lo] + (data[hi] - data[lo]) * (idx - lo)

        return {
            "n": n,
            "mean": statistics.mean(data),
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
            "min": data[0],
            "max": data[-1],
            "std": statistics.stdev(data) if n > 1 else 0.0,
        }

    def _build_summary_locked(self, since_count: int) -> dict:
        return {
            "total_completed": self._total_completed,
            "since_count": since_count,
            "queue_wait": self._percentiles(self._queue_wait_buf),
            "gen_time":   self._percentiles(self._gen_time_buf),
            "merge_time": self._percentiles(self._merge_time_buf),
            "e2e_time":   self._percentiles(self._e2e_time_buf),
        }

    def stats(self) -> dict:
        with self._lock:
            return {
                "window": self._window,
                "total_completed": self._total_completed,
                "inflight": len(self._inflight),
                "queue_wait_s": self._percentiles(self._queue_wait_buf),
                "gen_time_s":   self._percentiles(self._gen_time_buf),
                "merge_time_s": self._percentiles(self._merge_time_buf),
                "e2e_time_s":   self._percentiles(self._e2e_time_buf),
            }

    @staticmethod
    def _log_summary(s: dict) -> None:
        def fmt(d: dict) -> str:
            if d["n"] == 0:
                return "no data"
            return (
                f"n={d['n']:4d}  "
                f"mean={d['mean']:6.2f}  "
                f"p50={d['p50']:6.2f}  "
                f"p90={d['p90']:6.2f}  "
                f"p99={d['p99']:6.2f}  "
                f"[{d['min']:.2f}, {d['max']:.2f}]"
            )

        logger.info(
            f"\n"
            f"┌─ Rollout latency report "
            f"(total={s['total_completed']}, last_batch={s['since_count']}) ────────────\n"
            f"│  queue_wait : {fmt(s['queue_wait'])}\n"
            f"│  gen_time   : {fmt(s['gen_time'])}\n"
            f"│  merge_time : {fmt(s['merge_time'])}\n"
            f"│  e2e_time   : {fmt(s['e2e_time'])}\n"
            f"└──────────────────────────────────────────────────────────────────"
        )