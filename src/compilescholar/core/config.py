# -*- coding: utf-8 -*-
"""Run configuration: layered YAML -> one resolved, typed object written into every run directory.

Merge order (later wins): configs/base.yaml < configs/local.yaml (machine-local, gitignored) < the experiment config
passed with --config < --set key.path=value overrides on the command line. Dicts merge deeply, scalars replace.
Secrets never live here (they come from compilescholar.core.secrets).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from . import paths


@dataclass(frozen=True)
class AnswerConfig:
    model: str = "Qwen3.8-27B"
    kb: bool = True
    ext: bool = True
    cite: bool = False
    screen: bool = True
    state: bool = True
    probe: bool = True
    word_budget: int | None = None        # whole-answer word budget; None = no limit
    cutoff: str | None = None             # YYYY-MM knowledge cutoff (None = not filtered)
    kb_dir: str | None = None             # KB directory (relative paths resolve against the repo root)


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int = 3                      # questions answered in parallel
    local_max_concurrent: int = 48        # LLM lane width (LOCAL_MAX_CONCURRENT)
    sciverse_max_wait_s: int = 600        # queue for a Sciverse token instead of failing (SCIVERSE_MAX_WAIT_S)


@dataclass(frozen=True)
class BenchConfig:
    name: str = "cs2"                     # cs2 | dsb | multi108
    split: str = "dev"
    offset: int = 0
    limit: int = 100


@dataclass(frozen=True)
class RunConfig:
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def kb_path(self) -> Path | None:
        if not self.answer.kb or not self.answer.kb_dir:
            return None
        p = Path(self.answer.kb_dir)
        return p if p.is_absolute() else paths.REPO / p


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _read(p: Path) -> dict:
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return d


def _parse_set(items: list[str]) -> dict:
    out: dict = {}
    for it in items or []:
        key, _, raw = it.partition("=")
        if not key or not _:
            raise ValueError(f"--set expects key.path=value, got {it!r}")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def _build(cls, d: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kw = {}
    for name, f in known.items():
        if name not in d:
            continue
        v = d[name]
        sub = {"answer": AnswerConfig, "runtime": RuntimeConfig, "bench": BenchConfig}.get(name) if cls is RunConfig else None
        kw[name] = _build(sub, v or {}) if sub else v
    return cls(**kw)


def load(config: str | Path | None = None, overrides: list[str] | None = None) -> RunConfig:
    root = paths.REPO / "configs"
    merged = _deep_merge(_read(root / "base.yaml"), _read(root / "local.yaml"))
    if config:
        merged = _deep_merge(merged, _read(Path(config)))
    merged = _deep_merge(merged, _parse_set(overrides or []))
    merged.pop("description", None)
    return _build(RunConfig, merged)
