# -*- coding: utf-8 -*-
"""The one store of the diachronic-field-cognition system (DESIGN-UPGRADE §12a).

Layout: data/dfc/<stage>.sqlite, one file per stage, plus data/dfc/manifests/<stage>.json.
Each stage reads only the tables of the stages listed in UPSTREAM; raw inputs (parquet, HTML, OAI) are read by
`papers` and `citations` only.

Manifest = {stage, params, upstream: {stage: digest}, code: {file: sha256}, counts, digest, built_at}.
digest = sha256 over (stage, params, upstream digests, code hashes). A stage is stale when the digest it would have
now (current upstream digests, current code, same params) differs from the one on disk — so any change upstream or in
the stage's code invalidates everything downstream, and a rebuild is the only way to clear it."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from ..core import paths

STAGES = ("papers", "citations", "extract", "cognition")
UPSTREAM = {"papers": (), "citations": ("papers",), "extract": ("papers", "citations"),
            "cognition": ("papers", "citations", "extract")}
SRC = Path(__file__).resolve().parents[1]
CODE = {
    "papers": ("corpus/papers.py", "sources/arxiv_oai.py", "sources/arxiv_snapshot.py"),
    "citations": ("citations/markdown.py", "citations/html.py", "citations/resolve.py", "citations/build.py",
                  "compile/skeleton/bib.py", "compile/skeleton/resolve.py", "sources/arxiv_html.py"),
    "extract": ("extract/schema.py", "extract/self_pass.py", "extract/other_pass.py", "extract/build.py"),
    "cognition": ("cognition/",),
}


def root() -> Path:
    return paths.data() / "dfc"


def db_path(stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(stage)
    return root() / f"{stage}.sqlite"


def connect(stage: str, readonly: bool = False) -> sqlite3.Connection:
    p = db_path(stage)
    if readonly:
        return sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    # writers are shared by worker threads that serialize writes with their own lock
    con = sqlite3.connect(p, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _code_hashes(stage: str) -> dict[str, str]:
    out = {}
    for rel in CODE[stage]:
        p = SRC / rel
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            if f.exists():
                out[str(f.relative_to(SRC)).replace("\\", "/")] = hashlib.sha256(
                    f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return out


def _manifest_path(stage: str) -> Path:
    return root() / "manifests" / f"{stage}.json"


def read_manifest(stage: str) -> dict | None:
    p = _manifest_path(stage)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _digest(stage: str, params: dict, upstream: dict, code: dict) -> str:
    blob = json.dumps({"stage": stage, "params": params, "upstream": upstream, "code": code}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def upstream_digests(stage: str) -> dict[str, str]:
    out = {}
    for u in UPSTREAM[stage]:
        m = read_manifest(u)
        if m is None:
            raise RuntimeError(f"stage {stage!r} needs {u!r}, which has not been built")
        out[u] = m["digest"]
    return out


def write_manifest(stage: str, params: dict, counts: dict) -> dict:
    up = upstream_digests(stage)
    code = _code_hashes(stage)
    m = {"stage": stage, "params": params, "upstream": up, "code": code, "counts": counts,
         "digest": _digest(stage, params, up, code), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    p = _manifest_path(stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(p, "w", encoding="utf-8"), indent=1)
    return m


def status(stage: str) -> dict:
    """{'built': bool, 'stale': bool, 'why': [...]}: stale when upstream digests or code changed since the build."""
    m = read_manifest(stage)
    if m is None:
        return {"built": False, "stale": True, "why": ["not built"]}
    why = []
    for u in UPSTREAM[stage]:
        um = read_manifest(u)
        if um is None:
            why.append(f"upstream {u} missing")
        elif um["digest"] != m["upstream"].get(u):
            why.append(f"upstream {u} rebuilt")
        elif status(u)["stale"]:
            why.append(f"upstream {u} stale")
    code = _code_hashes(stage)
    changed = sorted(k for k in set(code) | set(m["code"]) if code.get(k) != m["code"].get(k))
    if changed:
        why.append("code changed: " + ", ".join(changed))
    return {"built": True, "stale": bool(why), "why": why}


def require_fresh(*stages: str) -> None:
    """Raise unless every given stage is built and not stale (called by a stage before it reads upstream tables)."""
    bad = {s: status(s)["why"] for s in stages if status(s)["stale"]}
    if bad:
        raise RuntimeError(f"upstream not fresh: {bad}")
