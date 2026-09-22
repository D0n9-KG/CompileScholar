# -*- coding: utf-8 -*-
"""Table extraction channel — ONE stage, two phases (A4 merge 2026-09-22).

The table channel was historically two separate stages with codenames
(F24 deterministic parse / F35 semantic) driven by hand. They process the
same data (mineru table structures) in sequence, so they are now ONE stage
entry: `table_extract`. The implementation modules stay intact (battle-
tested, 796 + 505 lines, each with its own tests) — this module is the
single CLI/stage surface:

  phase 1  parse    : table_channel  — deterministic table->records (zero LLM;
                    the PS-era lesson: LLM misreads table values, so tables
                    get a dedicated deterministic channel)
  phase 2  semantic : table_semantic — LLM lifts table cells into semantic
                    records (config/result axes) with the G5 canary gate

Usage (build.py drives this; standalone for debugging):
  python -m kb_compiler.records.table_extract \
      --texts DIR --cards C.json --registry R.json --vocab V.json \
      --checked records_checked.json --out kb/records_tables.json \
      --semantic-dir kb/table_semantic --model {MODEL} --provider local

  --phase parse    : run phase 1 only (equivalent to old table_channel)
  --phase semantic : run phase 2 only (equivalent to old table_semantic;
                     --records must be the phase-1 output)
  --phase both     : default — parse then semantic
"""
from __future__ import annotations

import argparse
import sys

from .common import load_json


def run_parse(args) -> str:
    """Phase 1 — delegate to table_channel's main logic with our args."""
    from . import table_channel as tc
    # table_channel's own CLI is preserved; here we call its main() through
    # a thin argv shim so the battle-tested entry stays single-sourced.
    import types
    ns = types.SimpleNamespace(
        texts=args.texts, checked=args.checked, cards=args.cards,
        registry=args.registry, out=args.out)
    return tc._run(ns) if hasattr(tc, "_run") else _shim_tc(ns)


def _shim_tc(ns):
    # fallback: drive via subprocess-free direct call of its main body is not
    # available; use module main with a patched argv
    import sys as _sys
    old_argv = _sys.argv
    try:
        _sys.argv = ["table_channel",
                     "--texts", ns.texts, "--checked", ns.checked,
                     "--out", ns.out,
                     "--cards", ns.cards, "--registry", ns.registry]
        from . import table_channel
        table_channel.main()
        return ns.out
    finally:
        _sys.argv = old_argv


def run_semantic(args) -> str:
    """Phase 2 — delegate to table_semantic (LLM, canary-gated)."""
    import sys as _sys
    old_argv = _sys.argv
    try:
        argv = ["table_semantic",
                "--records", args.out,
                "--texts", args.texts,
                "--registry", args.registry,
                "--out-dir", args.semantic_dir,
                "--provider", args.provider, "--model", args.model]
        if args.canary:
            argv.append("--canary")
        _sys.argv = argv
        from . import table_semantic
        table_semantic.main()
        return args.semantic_dir
    finally:
        _sys.argv = old_argv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--texts", required=True)
    ap.add_argument("--cards", default="")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--vocab", default="")
    ap.add_argument("--checked", required=True,
                    help="postcheck output (records_checked.json) — dedup base")
    ap.add_argument("--out", required=True,
                    help="phase-1 output (deterministic table records)")
    ap.add_argument("--semantic-dir", default="",
                    help="phase-2 output dir (default: <out-dir>/table_semantic)")
    ap.add_argument("--phase", choices=["parse", "semantic", "both"],
                    default="both")
    ap.add_argument("--provider", default="local")
    ap.add_argument("--model", default="local:Qwen3.8-27B",
                    help="LLM for phase 2 ONLY. The historical default was the "
                         "stale Qwen3.6 — pass explicitly (marathon-era lesson).")
    ap.add_argument("--canary", action="store_true",
                    help="run the G5 canary preflight before phase 2")
    args = ap.parse_args()
    if not args.semantic_dir:
        import os
        args.semantic_dir = os.path.join(os.path.dirname(args.out) or ".",
                                         "table_semantic")

    if args.phase in ("parse", "both"):
        print(f"[table_extract] phase 1/2 (parse, deterministic) -> {args.out}",
              flush=True)
        run_parse(args)
    if args.phase in ("semantic", "both"):
        print(f"[table_extract] phase 2/2 (semantic, LLM={args.model}) "
              f"-> {args.semantic_dir}", flush=True)
        run_semantic(args)
    print("[table_extract] done", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
