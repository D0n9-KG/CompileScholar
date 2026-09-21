# -*- coding: utf-8 -*-
"""Gate-3 attribution diagnostic: same 256 mentions, same Intern model,
LEGACY single-call merge path vs the blocked architecture.

If single-call also over-merges -> model-judgment cause (architecture
innocent; mitigation = prompts/guards/review). If single-call is clean ->
the blocked context (smaller prompts / recovery pass) caused it.

Usage: python gate3_singlecall_diag.py
"""
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.records.registry import collect_mentions, merge_entities  # noqa: E402

ARCH = (r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/"
        r"archive/paperscope_2026-09/paperscope_r2")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cards = json.load(open(os.path.join(ARCH, "cards_ps16.json"), encoding="utf-8"))
    legacy = json.load(open(os.path.join(ARCH, "registry_v32_pilot.json"), encoding="utf-8"))
    mentions = collect_mentions(cards)
    print(f"mentions: {len(mentions)} (single call, under 600 cap)")

    entities = merge_entities(mentions, cards, "intern:qwen3.8-27b")

    legacy_of = {}
    for e in legacy["entities"]:
        for a in e["aliases"]:
            legacy_of[a.strip().lower()] = e["entity_id"]
    new_of = {}
    for e in entities:
        for a in e["aliases"]:
            new_of[a.strip().lower()] = e["entity_id"]
    shared = sorted(set(new_of) & set(legacy_of))

    tp = fp = fn = 0
    over = []
    for a, b in combinations(shared, 2):
        ln = legacy_of[a] == legacy_of[b]
        nn = new_of[a] == new_of[b]
        if ln and nn:
            tp += 1
        elif nn and not ln:
            fp += 1
            over.append((a, b))
        elif ln and not nn:
            fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"SINGLE-CALL on intern: entities {len(entities)} | shared {len(shared)}")
    print(f"pair agreement vs legacy: P={prec:.4f} R={rec:.4f} F1={f1:.4f} "
          f"(tp={tp} over-merge={fp} split={fn})")
    print("over-merges:")
    for a, b in over[:30]:
        print(f"  {a[:45]!r} <-> {b[:45]!r}")
    json.dump({"mode": "single-call", "model": "intern:qwen3.8-27b",
               "entities": len(entities), "shared": len(shared),
               "P": prec, "R": rec, "F1": f1, "tp": tp, "over": fp, "split": fn,
               "over_pairs": over},
              open(os.path.join(HERE, "gate3_singlecall_diag.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
