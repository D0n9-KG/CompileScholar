"""Per-source circuit breaker (moved from sci-evo retrieval.circuit; FITNESS-LIBRARY-ACQUIRE: directly usable).

SourceCircuit: sliding-window failure tracker per source with CLOSED / OPEN / HALF_OPEN states. >= failure_threshold
errors inside window_s trips the source OPEN for cooldown_s; one probe request is allowed after the cooldown
(HALF_OPEN) and its success closes the circuit again. Only network errors and 5xx should be recorded as failures —
quota waits (429) belong to the rate limiter, and an honest empty result is not a failure. State is per process.
Deterministic, no LLM, thread-safe. (The tiered-discovery caller of the old package is retired.)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _SourceState:
    failures: list[float] = field(default_factory=list)  # timestamps
    opened_at: float | None = None
    probe_in_flight: bool = False


class SourceCircuit:
    def __init__(self, *, failure_threshold: int = 3, window_s: float = 120.0,
                 cooldown_s: float = 600.0):
        self.failure_threshold = failure_threshold
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._states: dict[str, _SourceState] = {}
        # RLock: snapshot() calls state() while holding the lock
        self._lock = threading.RLock()

    def _state(self, source: str) -> _SourceState:
        if source not in self._states:
            self._states[source] = _SourceState()
        return self._states[source]

    def state(self, source: str) -> str:
        """closed | open | half_open"""
        with self._lock:
            st = self._states.get(source)
            if st is None or st.opened_at is None:
                return "closed"
            if time.time() - st.opened_at >= self.cooldown_s:
                return "half_open"
            return "open"

    def allow(self, source: str) -> bool:
        """May we send a request to this source right now? A half_open
        source allows exactly one concurrent probe."""
        with self._lock:
            st = self._states.get(source)
            if st is None or st.opened_at is None:
                return True
            if time.time() - st.opened_at >= self.cooldown_s:
                if st.probe_in_flight:
                    return False  # one probe at a time
                st.probe_in_flight = True
                return True
            return False

    def record(self, source: str, ok: bool) -> None:
        with self._lock:
            st = self._state(source)
            if ok:
                st.failures.clear()
                st.opened_at = None
                st.probe_in_flight = False
                return
            now = time.time()
            st.failures = [t for t in st.failures if now - t <= self.window_s]
            st.failures.append(now)
            st.probe_in_flight = False
            if len(st.failures) >= self.failure_threshold:
                st.opened_at = now

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            out = {}
            for source in self._states:
                st = self._states[source]
                out[source] = {
                    "state": self.state(source),
                    "recent_failures": len(st.failures),
                }
            return out


# module-level default (single process; A3/A4 layers read .snapshot for the
# latency/status report)
DEFAULT_CIRCUIT = SourceCircuit()
