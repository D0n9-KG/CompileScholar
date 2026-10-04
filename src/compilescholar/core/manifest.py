# -*- coding: utf-8 -*-
"""Run manifests: everything needed to say what produced a run and to verify it on another machine.

Hashes are taken over LF-normalised content for text files (a Windows checkout with autocrlf would otherwise change
every hash) and raw bytes for binaries. `verify()` recomputes them.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from . import paths

_TEXT = {".py", ".txt", ".json", ".yaml", ".yml", ".md", ".jsonl", ".patch", ".toml"}
# env vars that change behaviour and are not secrets
_ENV_KEYS = ("KNOWLEDGE_CUTOFF", "LOCAL_MAX_CONCURRENT", "LOCAL_LARGE_MAX_CONCURRENT", "SCIVERSE_MAX_WAIT_S",
             "SCIVERSE_RATE_PER_MIN", "SCIVERSE_SHARED_BUCKET", "LLM_WALL_TIMEOUT", "LOCAL_SOCK_TIMEOUT",
             "LOCAL_MAX_ATTEMPTS", "LLM_SEED", "LLM_PROVIDER_ALLOWLIST", "ANSWER_MODEL", "EMBEDDING_MODEL",
             "CS_REFGRAPH_CACHE", "CS_TLS_INSECURE")


def file_hash(p: Path) -> str:
    data = Path(p).read_bytes()
    if Path(p).suffix.lower() in _TEXT:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git(*a) -> str:
    try:
        return subprocess.run(["git", "-C", str(paths.REPO), *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def package_files() -> list[Path]:
    root = paths.REPO / "src" / "compilescholar"
    return sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def build(run_id: str, config: dict, inputs: dict[str, Path], extra: dict | None = None) -> dict:
    """inputs: name -> file whose content the run depends on (KB files, question file, rubric...)."""
    from ..answer import prompts
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", "src", "configs", "pyproject.toml")
    return {
        "run_id": run_id,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git": {"head": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty_paths": dirty.splitlines() if dirty else []},
        "argv": sys.argv,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "config": config,
        "prompts_sha256": prompts.hashes(),
        "code_sha256": {str(p.relative_to(paths.REPO)).replace("\\", "/"): file_hash(p) for p in package_files()},
        "inputs_sha256": {k: file_hash(Path(v)) for k, v in inputs.items() if Path(v).exists()},
        "env": {k: os.environ[k] for k in _ENV_KEYS if k in os.environ},
        **(extra or {}),
    }


def write(run_dir: Path, manifest: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "manifest.json"
    json.dump(manifest, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


def verify(manifest_path: Path, inputs: dict[str, Path] | None = None) -> dict:
    """Recompute code and input hashes; returns {"ok": bool, "code_mismatch": [...], "input_mismatch": [...]}."""
    m = json.load(open(manifest_path, encoding="utf-8"))
    code_bad = [rel for rel, h in m.get("code_sha256", {}).items()
                if not (paths.REPO / rel).exists() or file_hash(paths.REPO / rel) != h]
    inp_bad = []
    for k, h in m.get("inputs_sha256", {}).items():
        p = (inputs or {}).get(k)
        if p is not None and (not Path(p).exists() or file_hash(Path(p)) != h):
            inp_bad.append(k)
    return {"ok": not code_bad and not inp_bad, "code_mismatch": code_bad, "input_mismatch": inp_bad}
