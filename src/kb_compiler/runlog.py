# -*- coding: utf-8 -*-
"""Structured run logging (P2-3): one JSONL event stream per pipeline run.

Replaces the print-to-stdout convention (123 print calls, zero logging) that
made incident forensics mean grepping 20 scattered .log files. Design:

  runlog.run("stage-name")  -> context manager wrapping one stage execution:
      writes events.jsonl into runs/<stage>-<ts>/ with start/finish/error
      events, plus a summary.json at completion. Every event carries the
      stage, wall clock, and a free-form payload dict.

  runlog.event(kind, **fields)  -> ad-hoc event inside a run
  runlog.counter(name, n=1)     -> cumulative counters, flushed into the
                                   finish event (papers, records, drops...)

Zero-config: if no run context is active, event() is a no-op (callers don't
need to care whether they're being run under the runner).

Usage:
    from kb_compiler import runlog
    with runlog.run("postcheck") as log:
        log.event("phase", name="gates", papers=len(papers))
        log.counter("records_checked", n)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager

_RUNS_DIR = os.environ.get(
    "KB_RUNS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "runs", "_legacy_build_manifests"))

_tls = threading.local()


def _active():
    return getattr(_tls, "run", None)


def _emit(rec: dict) -> None:
    run = _active()
    if run is None:
        return
    rec = {"ts": round(time.time(), 3), "run": run["name"], **rec}
    with run["lock"]:
        run["fh"].write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        run["fh"].flush()


def event(kind: str, **fields) -> None:
    """Ad-hoc event; no-op outside a run context."""
    _emit({"event": kind, **fields})


def counter(name: str, n: int = 1) -> None:
    run = _active()
    if run is None:
        return
    with run["lock"]:
        run["counters"][name] = run["counters"].get(name, 0) + n


class _RunCtx(dict):
    """Attribute access on the run context (log.event / log.counter sugar)."""
    def event(self, kind, **fields):
        event(kind, **fields)

    def counter(self, name, n=1):
        counter(name, n)


@contextmanager
def run(name: str):
    os.makedirs(_RUNS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(_RUNS_DIR, f"{name}-{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    fh = open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8")
    ctx = _RunCtx({"name": f"{name}-{stamp}", "dir": run_dir, "fh": fh,
                   "lock": threading.Lock(), "counters": {},
                   "t0": time.time(), "events": 0})
    _tls.run = ctx
    _emit({"event": "start", "pid": os.getpid(),
           "argv": sys.argv[:12]})
    err = None
    try:
        yield ctx
    except BaseException as e:
        err = e
        _emit({"event": "error", "error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        dt = time.time() - ctx["t0"]
        _emit({"event": "finish", "wall_s": round(dt, 1),
               "counters": ctx["counters"],
               "status": "error" if err else "ok"})
        summary = {"run": ctx["name"], "wall_s": round(dt, 1),
                   "status": "error" if err else "ok",
                   "counters": ctx["counters"]}
        with ctx["lock"]:
            fh.close()
        with open(os.path.join(run_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        _tls.run = None
