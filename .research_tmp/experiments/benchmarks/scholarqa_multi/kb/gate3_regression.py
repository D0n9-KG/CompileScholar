# -*- coding: utf-8 -*-
"""Acceptance gate ③: blocked-merge regression against the legacy PS registry.

Runs the PRODUCTION path (collect_mentions(cards_ps16) ->
merge_entities_blocked, local Qwen3.8-27B) on the 16-paper PaperScope corpus
and compares against registry_v32_pilot.json (campaign-audited reference).

Metrics on shared surfaces:
- pair precision/recall/F1 of same-entity judgments (new vs legacy)
- entity counts, singleton rate, cross-round merge counts
- divergence examples (top-mentioned surfaces merged/split vs legacy)

Interpretation note: legacy itself was LLM-built (different model era), so
divergence != error; this gate bounds the drift of the new architecture
against the last human-audited registry at small scale.

Usage: python gate3_regression.py [--model local:Qwen3.8-27B]
"""
import argparse
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")

from kb_compiler.records.registry import collect_mentions, merge_entities_blocked  # noqa: E402

ARCH = (r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/"
        r"archive/paperscope_2026-09/paperscope_r2")
CARDS16 = os.path.join(ARCH, "cards_ps16.json")
LEGACY = os.path.join(ARCH, "registry_v32_pilot.json")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="local:Qwen3.8-27B")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cards = json.load(open(CARDS16, encoding="utf-8"))
    legacy = json.load(open(LEGACY, encoding="utf-8"))
    mentions = collect_mentions(cards)
    print(f"cards: {len(cards)} | mentions (round-1 surfaces): {len(mentions)}")

    legacy_of = {}
    for e in legacy["entities"]:
        for a in e["aliases"]:
            legacy_of[a.strip().lower()] = e["entity_id"]

    entities, qc = merge_entities_blocked(
        mentions, cards, args.model,
        embed_cache_dir=os.path.join(HERE, "embed_cache"),
        tau1=0.84, tau2=0.80,
        log=lambda *a: print("  ", *a, flush=True))

    new_of = {}
    for e in entities:
        for a in e["aliases"]:
            new_of[a.strip().lower()] = e["entity_id"]

    shared = sorted(set(new_of) & set(legacy_of))
    print(f"entities: new {len(entities)} vs legacy {len(legacy['entities'])} "
          f"| shared surfaces {len(shared)} "
          f"(new {len(new_of)}, legacy {len(legacy_of)})")

    # pair metrics over shared surfaces
    tp = fp = fn = 0
    divergences = []
    mention_n = {k: len(v["papers"]) for k, v in mentions.items()}
    for a, b in combinations(shared, 2):
        ln = legacy_of[a] == legacy_of[b]
        nn = new_of[a] == new_of[b]
        if ln and nn:
            tp += 1
        elif nn and not ln:
            fp += 1
            divergences.append(("over-merge", a, b, mention_n.get(a, 0) + mention_n.get(b, 0)))
        elif ln and not nn:
            fn += 1
            divergences.append(("split", a, b, mention_n.get(a, 0) + mention_n.get(b, 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"pair agreement vs legacy: P={prec:.4f} R={rec:.4f} F1={f1:.4f} "
          f"(tp={tp} over-merge={fp} split={fn})")
    singletons = sum(1 for e in entities if len(e["aliases"]) == 1)
    print(f"new: singletons {singletons}/{len(entities)} | qc: {json.dumps(qc, ensure_ascii=False)[:400]}")

    divergences.sort(key=lambda d: -d[3])
    print(f"\ntop divergences by mention weight ({len(divergences)} total):")
    for kind, a, b, w in divergences[:20]:
        print(f"  [{kind}] w={w} {a[:40]!r} <-> {b[:40]!r}")

    out = os.path.join(HERE, "gate3_regression_report.json")
    json.dump({"model": args.model, "n_cards": len(cards),
               "n_mentions": len(mentions), "n_entities_new": len(entities),
               "n_entities_legacy": len(legacy["entities"]),
               "shared_surfaces": len(shared),
               "pair_precision": prec, "pair_recall": rec, "pair_f1": f1,
               "tp": tp, "over_merge": fp, "split": fn,
               "singletons": singletons, "qc": qc,
               "divergences_top": [{"kind": k, "a": a, "b": b, "w": w}
                                   for k, a, b, w in divergences[:60]]},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
