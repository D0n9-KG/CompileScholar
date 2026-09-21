# -*- coding: utf-8 -*-
"""Gate ③ (revised): architecture A/B equivalence, NOT legacy-registry match.

The original gate-3 comparison was invalidated (evidence 09-21): legacy
registry_v32_pilot was built from DIFFERENT cards (KB v3.2 re-extraction,
clean method_identity) than cards_ps16.json (old protocol, stuffed aliases).
The 27 "over-merges" were own-paper canonical-collapse artifacts (all
members=1 — the LLM merged none of them), present in ANY architecture using
these cards.

Revised gate: same cards_ps16 mentions, same model, LEGACY single-call path
(A) vs BLOCKED path (B). Both apply identical own-override/collapse logic, so
artifacts cancel in the comparison. Pass criteria:
- B's alias-pair agreement with A: F1 >= 0.90 (architecture equivalence)
- B introduces no large split direction (recall >= 0.95: blocking must not
  separate what the single call merges — the slice-isolation defense)

Usage: python gate3_ab_architecture.py [--model local:Qwen3.8-27B]
"""
import argparse
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.records.registry import (collect_mentions, merge_entities,
                                          merge_entities_blocked)

ARCH = (r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/"
        r"archive/paperscope_2026-09/paperscope_r2")
HERE = os.path.dirname(os.path.abspath(__file__))


def pair_map(entities):
    of = {}
    for e in entities:
        for a in e["aliases"]:
            of[a.strip().lower()] = e["entity_id"]
    return of


def compare(a_of, b_of, la, lb):
    shared = sorted(set(a_of) & set(b_of))
    tp = fp = fn = 0
    over, splits = [], []
    for x, y in combinations(shared, 2):
        an = a_of[x] == a_of[y]
        bn = b_of[x] == b_of[y]
        if an and bn:
            tp += 1
        elif bn and not an:
            fp += 1
            over.append((x, y))
        elif an and not bn:
            fn += 1
            splits.append((x, y))
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    print(f"{lb} vs {la}: shared {len(shared)} | P={P:.4f} R={R:.4f} F1={F1:.4f} "
          f"(agree={tp} b-extra-merge={fp} b-split={fn})")
    for x, y in over[:10]:
        print(f"   b-extra-merge: {x[:40]!r} + {y[:40]!r}")
    for x, y in splits[:10]:
        print(f"   b-SPLIT: {x[:40]!r} + {y[:40]!r}")
    return {"shared": len(shared), "P": P, "R": R, "F1": F1,
            "b_extra_merge": fp, "b_split": fn,
            "extra_pairs": over[:40], "split_pairs": splits[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="local:Qwen3.8-27B")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cards = json.load(open(os.path.join(ARCH, "cards_ps16.json"), encoding="utf-8"))
    mentions = collect_mentions(cards)
    print(f"mentions: {len(mentions)} | model: {args.model}")

    print("\n== A: legacy single call ==")
    ents_a = merge_entities(mentions, cards, args.model)
    print(f"A: {len(ents_a)} entities")

    print("\n== B: blocked (tau1=0.84, tau2=0.80, recovery on) ==")
    ents_b, qc = merge_entities_blocked(
        mentions, cards, args.model,
        embed_cache_dir=os.path.join(HERE, "embed_cache"),
        tau1=0.84, tau2=0.80, log=lambda *a: print("  ", *a, flush=True))
    print(f"B: {len(ents_b)} entities | qc merges: blocks={qc['block_merges']} "
          f"recovery={qc['singleton_recovery'].get('recovered')} "
          f"cross={[r['merges'] for r in qc['cross_rounds']]}")

    res = compare(pair_map(ents_a), pair_map(ents_b), "A(single)", "B(blocked)")
    res["qc_b"] = qc
    res["model"] = args.model
    res["n_entities"] = {"A": len(ents_a), "B": len(ents_b)}
    out = os.path.join(HERE, "gate3_ab_report.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nreport -> {out}")
    ok = res["F1"] >= 0.90 and res["R"] >= 0.95
    print(f"GATE ③ (revised): {'PASS' if ok else 'FAIL'} "
          f"(F1 {res['F1']:.3f} >= 0.90, R {res['R']:.3f} >= 0.95)")


if __name__ == "__main__":
    main()
