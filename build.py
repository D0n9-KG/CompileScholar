#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build.py — single pipeline entry (A4 part 3, 2026-09-22).

Design conventions (researched 2026-09-22 from Snakemake/Kedro/Dagster/DVC —
see .research_tmp/scratch/agent_research_final.md for sources):
  1. stages declared as DATA (a STAGES registry), not as code in main()
  2. outputs are files at fixed paths; a stage is fresh iff its outputs exist
     AND the manifest's recorded input/param/code hashes match
  3. LLM stages (code_sensitive=False) NEVER auto-rerun on code change —
     output-present = skip; rerun requires --force (Snakemake protect() semantics)
  4. every stage gets a cheap structural check (Dagster asset-check idea)
  5. every complete run writes runs/<ts>/manifest.json (DVC lock / ACM
     Functional-badging idea): git sha, config snapshot, per-stage hashes
  6. --list / --dry-run / --dot for pipeline introspection (Snakemake habits)

Deliberately NOT built (single-person research project, per the research
verdict): general DAG scheduler, plugin discovery, web UI, remote execution,
DataCatalog abstractions, multi-env matrices.

Stage execution uses run_stage.py semantics under the hood (env projection,
allowlist gate, purity assertion) — the existing battle-tested wrapper.
This file replaces the hand-driven chain of run_stage invocations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from kb_compiler.config import load_conf, project_env  # noqa: E402

MULTI = REPO / ".research_tmp/experiments/benchmarks/scholarqa_multi"
KB = MULTI / "kb"


# ---------------------------------------------------------------------------
# Stage registry — module name = stage name = product name (four-name unity)
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str                      # "deep_extract", "postcheck", ...
    desc: str                      # human-readable one-liner
    cmd: list                      # argv template; "{...}" fields from conf
    outputs: tuple[str, ...]       # fixed product paths (relative to REPO)
    llm: bool = False              # True: output-present = skip (protect())
    check: str = ""                # "module:function" structural check
    params: tuple[str, ...] = ()   # config keys (dotted) that invalidate


def _p(rel: str) -> str:
    return str((REPO / rel).resolve())


STAGES = [
    Stage(
        name="cards",
        desc="identity cards from full texts (section labels, method identity)",
        cmd=["python", "-m", "kb_compiler.records.cards",
             "--texts", "{texts}", "--manifest", "{manifest}",
             "--out", str(KB / "cards.json"), "--model", "{model}"],
        outputs=("conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/cards.json",),
        llm=True,
    ),
    Stage(
        name="registry_vocab",
        desc="entity merge round 1 + dimension vocabulary (one registry.py run "
             "produces both registry.json and dim_vocab_v1.json into --out-dir)",
        cmd=["python", "-m", "kb_compiler.records.registry",
             "--cards", str(KB / "cards.json"), "--manifest", "{manifest}",
             "--out-dir", str(KB), "--merge-mode", "blocked",
             "--tau1", "{registry.tau1}", "--model", "{model}"],
        outputs=("conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/registry.json",
                 "conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/dim_vocab_v1.json"),
        llm=True,
    ),
    Stage(
        name="deep_extract",
        desc="global chunk-pool deep extraction (records_slot.json + WAL)",
        cmd=["python", "-m", "kb_compiler.records.deep_extract",
             "--texts", "{texts}", "--manifest", "{manifest}",
             "--cards", str(KB / "cards.json"),
             "--registry", str(KB / "registry.json"),
             "--vocab", str(KB / "dim_vocab_v1.json"),
             "--out", str(KB / "records_slot.json"),
             "--model", "{model}", "--pool", "{slot.pool_size}"],
        outputs=("conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/records_slot.json",),
        llm=True,
    ),
    Stage(
        name="postcheck",
        desc="five deterministic quality gates + repair-or-drop",
        cmd=["python", "-m", "kb_compiler.records.postcheck",
             "--records", str(KB / "records_slot.json"),
             "--texts", "{texts}", "--vocab", str(KB / "dim_vocab_v1.json"),
             "--out-dir", str(KB / "postcheck"), "--model", "{model}"],
        outputs=("conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/postcheck/records_checked.json",),
        llm=True,
    ),
    Stage(
        name="table_extract",
        desc="tables: deterministic parse (phase 1) + LLM semantic (phase 2)",
        cmd=["python", "-m", "kb_compiler.records.table_extract",
             "--texts", "{texts}", "--cards", str(KB / "cards.json"),
             "--registry", str(KB / "registry.json"),
             "--checked", str(KB / "postcheck/records_checked.json"),
             "--out", str(KB / "records_tables.json"),
             "--semantic-dir", str(KB / "table_semantic"),
             "--provider", "local", "--model", "{model}", "--canary"],
        outputs=("conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/records_tables.json",),
        llm=True,
    ),
    Stage(
        name="notation",
        desc="formula/notation symbol harvest",
        cmd=["python", "-m", "kb_compiler.records.notation_harvest",
             "--texts", "{texts}", "--records", str(KB / "records_tables_by_paper.json"),
             "--out", str(KB / "notation.json"),
             "--provider", "local", "--model", "{model}"],
        outputs=("conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/notation.json",),
        llm=True,
    ),
    Stage(
        name="registry_growth",
        desc="fold deep-extract entity_queue into registry (round 2)",
        cmd=["python", "-m", "kb_compiler.records.registry_growth",
             "--registry", str(KB / "registry.json"),
             "--records", str(KB / "records_slot.json"),
             "--out-dir", str(KB), "--model", "{model}"],
        outputs=("conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/registry_v2.json",),
        llm=True,
    ),
    Stage(
        name="views",
        desc="compile the four views from final records + grown registry",
        cmd=["python", "-m", "kb_compiler.views.compiler",
             "--records", str(KB / "records_slot.json"),
             "--registry", str(KB / "registry_v2.json"),
             "--vocab", str(KB / "dim_vocab_v1.json"),
             "--manifest", "{manifest}",
             "--out", str(KB / "views.json")],
        outputs=("conf/../.research_tmp/experiments/benchmarks/scholarqa_multi/kb/views.json",),
        llm=False,
    ),
]


# ---------------------------------------------------------------------------
# make layer: freshness, execution, checks
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def git_state() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10
                             ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, timeout=10
                               ).stdout.strip().splitlines()
        return {"sha": sha, "dirty_files": dirty[:50]}
    except Exception:
        return {"sha": "unknown", "dirty_files": []}


def _outputs_abs(st: Stage) -> list[Path]:
    return [REPO / o.replace("conf/../", "") for o in st.outputs]


def _format_cmd(st: Stage, cfg: dict) -> list[str]:
    fields = {
        "texts": cfg["paths"]["texts"],
        "manifest": cfg["paths"]["manifest"],
        "model": cfg["llm"]["model"],
        "slot.pool_size": str(cfg["slot"]["pool_size"]),
        "registry.tau1": str(cfg["registry"]["tau1"]),
    }
    out = []
    for a in st.cmd:
        for k, v in fields.items():
            a = a.replace("{%s}" % k, v)
        out.append(a)
    return out


def _param_fingerprint(st: Stage, cfg: dict) -> str:
    h = hashlib.sha256()
    for key in st.params:
        node = cfg
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        h.update(json.dumps(node, sort_keys=True, default=str).encode())
    h.update(json.dumps(_format_cmd(st, cfg)).encode())
    return h.hexdigest()[:16]


def _code_fingerprint(st: Stage) -> str:
    mod = st.cmd[st.cmd.index("-m") + 1] if "-m" in st.cmd else ""
    path = REPO / "src" / (mod.replace(".", "/") + ".py")
    return sha256_file(path) if path.exists() else "n/a"


def stage_fresh(st: Stage, cfg: dict, manifest: dict) -> bool:
    outs = _outputs_abs(st)
    if not all(o.exists() for o in outs):
        return False
    rec = manifest.get("stages", {}).get(st.name)
    if not rec:
        return False
    if st.llm:
        # protect() semantics: output present + manifest record = fresh;
        # code changes NEVER auto-rerun a GPU stage
        return True
    if rec.get("param_fp") != _param_fingerprint(st, cfg):
        return False
    if rec.get("code_fp") != _code_fingerprint(st):
        return False
    return True


def run_check(st: Stage) -> tuple[bool, str]:
    """Cheap structural check (Dagster asset-check idea). Unknown outputs
    (e.g. dir-shaped products) -> check = 'shape unknown' (pass)."""
    if not st.check:
        return True, "no check declared"
    mod_name, fn = st.check.split(":")
    mod = importlib.import_module(mod_name)
    ok, msg = getattr(mod, fn)()
    return ok, msg


def build(targets: list[str], force: set[str], dry: bool, exp: str | None):
    cfg = load_conf(exp=exp)
    manifest_path = REPO / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else {"stages": {}}

    stages = [s for s in STAGES if not targets or s.name in targets]
    for t in targets:
        if t not in {s.name for s in STAGES}:
            print(f"unknown stage: {t} (see --list)", file=sys.stderr)
            sys.exit(2)

    ran, skipped = [], []
    for st in stages:
        if st.name in force:
            fresh = False
        else:
            fresh = stage_fresh(st, cfg, manifest)
        status = "skip(fresh)" if fresh else ("FORCE" if st.name in force else "run")
        print(f"[{st.name:16s}] {status:12s} {st.desc}")
        if dry or fresh:
            skipped.append(st.name)
            continue
        cmd = _format_cmd(st, cfg)
        env = project_env(cfg)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["LLM_CALL_LOG"] = str(KB / "ledger_build.jsonl")
        env["LLM_RUN_ID"] = f"multi-{st.name}"
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(REPO / "src"), env=env)
        dt = time.time() - t0
        outs = _outputs_abs(st)
        if p.returncode != 0:
            print(f"[{st.name}] FAILED rc={p.returncode} — STOP "
                  f"(resume-safe; fix then rerun build.py)", file=sys.stderr)
            sys.exit(p.returncode)
        ok, msg = run_check(st)
        manifest["stages"][st.name] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "param_fp": _param_fingerprint(st, cfg),
            "code_fp": _code_fingerprint(st),
            "output_hashes": [sha256_file(o) if o.exists() else None for o in outs],
            "elapsed_s": round(dt, 1), "check": {"ok": ok, "msg": msg},
        }
        manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                 encoding="utf-8")
        ran.append(st.name)
        print(f"[{st.name:16s}] done {dt:.0f}s | check {'PASS' if ok else 'FAIL: ' + msg}")

    run_dir = REPO / "runs"
    run_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (run_dir / f"manifest-{stamp}.json").write_text(
        json.dumps({"git": git_state(), "config": cfg, "stages": manifest["stages"],
                    "ran": ran, "skipped": skipped}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nbuild: ran={ran or '-'} skipped={skipped or '-'} "
          f"| run manifest: runs/manifest-{stamp}.json")


def list_stages(exp: str | None):
    cfg = load_conf(exp=exp)
    manifest_path = REPO / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else {"stages": {}}
    print(f"{'stage':18s} {'llm':4s} {'fresh':6s} outputs")
    for st in STAGES:
        fresh = stage_fresh(st, cfg, manifest)
        outs = ", ".join(o.split("/")[-1] for o in st.outputs)
        print(f"{st.name:18s} {'LLM' if st.llm else 'det':4s} "
              f"{'yes' if fresh else 'no':6s} {outs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("targets", nargs="*", help="stage names (default: all)")
    ap.add_argument("--list", action="store_true", help="list stages + freshness")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", default="", help="comma-separated stages to force-rerun")
    ap.add_argument("--exp", default=None, help="conf/experiments/<name>.yaml delta")
    args = ap.parse_args()
    force = {s for s in args.force.split(",") if s}
    if args.list:
        list_stages(args.exp)
        return
    build(args.targets, force, args.dry_run, args.exp)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
