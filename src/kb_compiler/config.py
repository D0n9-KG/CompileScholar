# -*- coding: utf-8 -*-
"""Layered config loader (Hydra 思想的 30 行版，调研 2026-09-22).

Layers (low → high): conf/base.yaml < conf/local.yaml < conf/experiments/<name>.yaml
< env vars (secrets & one-off overrides, via .env / shell).

- dict 深合并，标量覆盖；列表整体替换。
- conf/local.yaml 与 conf/experiments/ 不存在时静默跳过（local 在 .gitignore）。
- 不装 Hydra/OmegaConf：单人单机场景，deep-merge + yaml 足够（调研结论：
  Hydra 的 decorator/launch/app 是为多 job sweep 设计的，不抄）。
- secrets 永远不进本层：API key/端点走 kb_infra.llm 的 .env 加载，职责分离。

Usage:
    from kb_compiler.config import load_conf
    cfg = load_conf()                # base + local
    cfg = load_conf("multi")         # + conf/experiments/multi.yaml delta
    cfg = load_conf(exp="multi", cli_overrides={"llm.model": "local:Qwen3.8-27B"})
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF = _REPO_ROOT / "conf"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(d).__name__}")
    return d


def load_conf(exp: str | None = None, cli_overrides: dict | None = None) -> dict:
    """Merged config: base < local < experiments/<exp> < cli_overrides."""
    cfg = _read_yaml(_CONF / "base.yaml")
    cfg = _deep_merge(cfg, _read_yaml(_CONF / "local.yaml"))
    if exp:
        cfg = _deep_merge(cfg, _read_yaml(_CONF / "experiments" / f"{exp}.yaml"))
    if cli_overrides:
        # dotted-path overrides: {"llm.model": "x"} -> {"llm": {"model": "x"}}
        flat: dict = {}
        for dotted, v in cli_overrides.items():
            node = flat
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        cfg = _deep_merge(cfg, flat)
    # resolve repo-relative paths to absolute
    for key, p in list(cfg.get("paths", {}).items()):
        if isinstance(p, str):
            cfg["paths"][key] = str((_REPO_ROOT / p).resolve()) if not os.path.isabs(p) else p
    return cfg


def project_env(cfg: dict, env: dict | None = None) -> dict:
    """Project the llm section onto kb_infra's env-var interface.

    kb_infra.llm 的读取逻辑（生产已验证）保持 env-var 接口不动；本函数是
    config → env 的唯一投影点（env 名错位 bug 的结构性解药：名字只在这里
    出现一次）。
    """
    e = dict(os.environ) if env is None else dict(env)
    llm = cfg.get("llm", {})
    mapping = {
        "LOCAL_MAX_CONCURRENT": llm.get("local_max_concurrent"),
        "LOCAL_SOCK_TIMEOUT": llm.get("local_sock_timeout"),
        "LOCAL_MAX_ATTEMPTS": llm.get("local_max_attempts"),
        "LLM_WALL_TIMEOUT": llm.get("wall_timeout"),
        "LLM_SOCK_TIMEOUT": llm.get("sock_timeout"),
        "LLM_PROVIDER_ALLOWLIST": llm.get("provider_allowlist"),
        "INTERN_SOFT_CAP": llm.get("intern_soft_cap"),
    }
    for k, v in mapping.items():
        if v is not None:
            e[k] = str(v)
    slot = cfg.get("slot", {})
    if slot.get("max_chars") is not None:
        e["SLOT_MAX_CHARS"] = str(slot["max_chars"])
    reg = cfg.get("registry", {})
    if reg.get("block_workers") is not None:
        e["REGISTRY_BLOCK_WORKERS"] = str(reg["block_workers"])
    return e
