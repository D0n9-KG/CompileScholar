# -*- coding: utf-8 -*-
"""Granular-flow second-domain pilot runner (GRANULAR-PILOT-PREREG.md v1).

Full pipeline on the 8-paper granular rheology family + granular canary:
skeleton -> registry/vocab (INDEPENDENT domain namespace, canary excluded)
-> slot -> postcheck -> canary scoring -> pre-registered metrics report.

Usage:
  python -m kb_compiler.records.run_granular_pilot \
      --base ../.research_tmp/experiments/stageB/granular_pilot \
      --model Qwen3.8-Max
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import registry as reg
from .canary_granular import CANARY_PID, CANARY_TEXT, score as canary_score
from .common import load_corpus, load_json, load_manifest, save_json
from .postcheck import run_postcheck
from .skeleton import build_card
from .slot import extract_paper

_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="Qwen3.8-Max")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    B = args.base.rstrip("/")
    t0 = time.time()

    texts = dict(load_corpus(f"{B}/texts"))
    manifest = load_manifest(f"{B}/manifest_granular8.json")
    texts[CANARY_PID] = CANARY_TEXT
    pids = sorted(texts)
    print(f"pilot: {len(pids)} papers (incl. canary), model={args.model}", flush=True)

    # ---- Stage 1: skeleton (8 papers + canary card) ----
    cards = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_card, pid, texts[pid],
                          (manifest.get(pid) or {}).get("title") or
                          "PSR: Pressure-Scaled Rheology for Dense Granular Flows",
                          args.model): pid for pid in pids}
        for f in futs:
            pid, card = f.result()
            if card:
                cards[pid] = card
    save_json(cards, f"{B}/paper_cards.json")
    print(f"skeleton: {len(cards)}/{len(pids)} cards", flush=True)

    # ---- Stage 1.5: registry + vocab (INDEPENDENT namespace; canary excluded) ----
    real_cards = {k: v for k, v in cards.items() if k != CANARY_PID}
    mentions = reg.collect_mentions(real_cards)
    entities = reg.merge_entities(mentions, real_cards, args.model)
    surface_index = {}
    for e in entities:
        for a in e["aliases"]:
            surface_index[reg._norm(a)] = e["entity_id"]
    registry = {"version": 1, "model": args.model, "domain": "granular_rheology",
                "entities": entities, "surface_index": surface_index}
    save_json(registry, f"{B}/registry.json")
    vocab = reg.build_vocab(real_cards, args.model)
    vocab["version"] = 1
    vocab["domain"] = "granular_rheology"
    vocab, flags = reg.clean_vocab(vocab)
    qc = reg.qc_report(vocab, registry)
    save_json(vocab, f"{B}/dim_vocab_v1.json")
    save_json({"cleanup_flags": flags, "qc": qc}, f"{B}/vocab_qc.json")
    types = Counter(e["entity_type"] for e in entities)
    print(f"registry: {len(entities)} entities {dict(types)} | vocab: "
          f"subject={len(vocab['subject'])} setup={len(vocab['setup'])} "
          f"variant={len(vocab['variant'])} hparam={len(vocab['hyperparam_items'])}",
          flush=True)

    # ---- Stage 2: slot ----
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_paper, pid, texts[pid], cards.get(pid) or {},
                          registry, vocab, args.model,
                          (manifest.get(pid) or {}).get("title") or
                          "PSR: Pressure-Scaled Rheology for Dense Granular Flows"): pid
                for pid in pids if pid in cards}
        for f in futs:
            pid, out = f.result()
            results[pid] = out
    save_json(results, f"{B}/records_raw.json")

    # ---- Stage 3: postcheck (live repair) ----
    checked, dropped, warnings, stats = run_postcheck(results, texts, vocab, args.model)
    save_json(checked, f"{B}/records_checked.json")
    save_json(dropped, f"{B}/dropped.json")
    save_json(warnings, f"{B}/warnings.json")

    # ---- canary + pre-registered metrics ----
    can = canary_score((checked.get(CANARY_PID) or {}).get("records", []))
    save_json(can, f"{B}/canary_score.json")
    n = max(1, stats.get("total", 0))
    ovf = stats.get("overflow", 0)
    real_checked = {k: v for k, v in checked.items() if k != CANARY_PID}
    relation_use = Counter(r.get("relation") for p in real_checked.values()
                           for r in p["records"] if r["kind"] == "lineage")
    dims_new = sum(len(r.get("dims_new") or {}) for p in real_checked.values()
                   for r in p["records"])
    queue = sum(len(p.get("entity_queue", [])) for p in results.values() if p)
    overflow_reasons = [o.get("reason", "") for p in real_checked.values()
                        for o in p.get("overflow", [])]
    report = {
        "model": args.model, "wall_s": round(time.time() - t0, 1),
        "rates": {"first_pass": round(stats.get("first_pass", 0) / n, 3),
                  "repair": round(stats.get("repaired", 0) / n, 3),
                  "drop": round(stats.get("dropped", 0) / n, 3),
                  "residual_overflow_reported_only": round(ovf / max(1, n + ovf), 3)},
        "canary": can,
        "relation_usage": dict(relation_use),
        "dims_new_count": dims_new, "entity_queue_count": queue,
        "overflow_reasons_raw": overflow_reasons,
        "stats": stats,
        "prereg_gates": {
            "first_pass>=0.60": report_gate(stats.get("first_pass", 0) / n >= 0.60),
            "drop<=0.15": report_gate(stats.get("dropped", 0) / n <= 0.15),
            "canary_recall>=4": report_gate(sum(can["facts"].values()) >= 4),
            "canary_traps==0": report_gate(can["trap_fired"] == 0),
            "frozen_layer_falsification": "PENDING manual overflow-cluster review (判读1)",
        },
    }
    save_json(report, f"{B}/pilot_report.json")
    print(json.dumps({k: report[k] for k in ("rates", "relation_usage",
                                             "dims_new_count", "entity_queue_count",
                                             "prereg_gates")},
                     ensure_ascii=False, indent=1), flush=True)
    print(f"canary: {can['fact_recall']} traps={can['trap_fired']}", flush=True)


def report_gate(ok):
    return "PASS" if ok else "FAIL"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
