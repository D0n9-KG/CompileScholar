# -*- coding: utf-8 -*-
"""Repository-relative paths. Every location the package reads or writes resolves here; nothing else hardcodes a path.

Roots (override with CS_ROOT, CS_DATA, CS_CACHE, CS_RUNS on another machine):
  data/library/        authoritative store (registry, parsed full texts) — not rebuildable, backed up
  data/derived/        derived store (stage files, manifests) — rebuildable from library + code
  data/benchmarks/<b>/ benchmark inputs (questions, rubrics, frozen KBs, archived arms); MANIFEST.tsv tracked in git
  data/external/<d>/   third-party datasets (ScholarCatalyst, IdeaForecast, ...)
  cache/<sub>/         rebuildable caches (LLM responses, refgraph, pace state)
  runs/<run_id>/       working runs (ignored); frozen runs are promoted to results/<name>/ (tracked)
  third_party/<name>/  pinned upstream checkouts (ignored; README + patches tracked)
Machine resources outside the repository (NAS snapshot, PDF mirrors, Sci-Hub archive, MinerU, harness cwd) are named
in configs/local.yaml under `paths:` and read with resource(key) — never a hardcoded default.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO = Path(os.environ.get("CS_ROOT") or Path(__file__).resolve().parents[3])


def data() -> Path:
    return Path(os.environ.get("CS_DATA") or REPO / "data")


def cache() -> Path:
    return Path(os.environ.get("CS_CACHE") or REPO / "cache")


def runs() -> Path:
    return Path(os.environ.get("CS_RUNS") or REPO / "runs")


def library() -> Path:
    return data() / "library"


def derived() -> Path:
    return data() / "derived"


def benchmarks(name: str) -> Path:
    return data() / "benchmarks" / name


def external(name: str) -> Path:
    return data() / "external" / name


def results(name: str = "") -> Path:
    return REPO / "results" / name if name else REPO / "results"


def third_party(name: str) -> Path:
    return REPO / "third_party" / name


def env_file() -> Path:
    return REPO / ".env"


# ---- machine resources (configs/local.yaml `paths:`)

_ENV = {"arxiv_snapshot": "CS_ARXIV_SNAPSHOT", "harness_cwd": "HARNESS_CWD", "astabench": "CS_ASTABENCH",
        "deepscholar_bench": "CS_DSB"}


@lru_cache(maxsize=1)
def _local_paths() -> dict:
    import yaml
    out: dict = {}
    for f in (REPO / "configs" / "base.yaml", REPO / "configs" / "local.yaml"):
        if f.exists():
            out.update((yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("paths") or {})
    return out


def resource(key: str) -> Path:
    """A location outside the repository, from configs/local.yaml `paths:` (an environment override per key is
    honoured, see _ENV). Unconfigured -> error naming the key; there is no hardcoded fallback."""
    v = os.environ.get(_ENV.get(key, "")) if key in _ENV else None
    v = v or _local_paths().get(key)
    if not v:
        raise RuntimeError(f"resource {key!r} is not configured: add `paths: {{{key}: ...}}` to configs/local.yaml "
                           f"(see configs/local.example.yaml)")
    return Path(v)


def has_resource(key: str) -> bool:
    try:
        resource(key)
        return True
    except RuntimeError:
        return False
