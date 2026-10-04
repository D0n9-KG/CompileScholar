# -*- coding: utf-8 -*-
"""Repository-relative paths. Every location the package reads or writes resolves here; nothing else hardcodes a path.

Override any root with an environment variable (useful on another machine): CS_ROOT, CS_DATA, CS_CACHE, CS_RUNS.
Until the W6 S8 data move, the KB and benchmark files still live under .research_tmp/experiments/benchmarks;
`legacy_bench()` points there and is the only place that knows it.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ.get("CS_ROOT") or Path(__file__).resolve().parents[3])


def data() -> Path:
    return Path(os.environ.get("CS_DATA") or REPO / "data")


def cache() -> Path:
    return Path(os.environ.get("CS_CACHE") or REPO / "cache")


def runs() -> Path:
    return Path(os.environ.get("CS_RUNS") or REPO / "runs")


def legacy_bench() -> Path:
    """Benchmark tree as it exists before the S8 data move."""
    return REPO / ".research_tmp" / "experiments" / "benchmarks"


def env_file() -> Path:
    return REPO / ".env"
