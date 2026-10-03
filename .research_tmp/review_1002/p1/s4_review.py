# -*- coding: utf-8 -*-
"""Dump refuted/unclear claims compactly for manual adjudication; then compute tables from manual_review.json."""
import json
import math
import os
import sys
from collections import Counter, defaultdict

P1 = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))


def dump(fn="s3_verified.json", key="claim"):
    out = json.load(open(os.path.join(P1, fn), encoding="utf-8"))
    for i, o in enumerate(out):
        if o.get("verdict") not in ("refuted", "unclear", "already_done"):
            continue
        print(f"### [{i}] {o.get('sysg', '')} {o.get('label', '')} {o.get('verdict')} framing={o.get('framing')} "
              f"date={o.get('claim_date') or o.get('year')}")
        print("CLAIM:", o.get(key) or o.get("missing"))
        print("SENT:", (o.get("sentence") or o.get("subject") or "")[:350])
        print("CX:", o.get("counterexample"))
        for r in o.get("refuting", []):
            print(f"  -> ({r['year']}) q{r['q_idx']} r{r['rank']} sy={r['same_year']} {r['title']}")
            print("     ABS:", (r.get("abstract") or "")[:600].replace("\n", " "))
        print()


if __name__ == "__main__":
    dump(*(sys.argv[1:] or []))
