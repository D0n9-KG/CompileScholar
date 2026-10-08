# -*- coding: utf-8 -*-
"""Stratified question-blind pilot sampling for SC.

Selection may only use STRUCTURAL fields (type, paper_domain) — never the question text, never the rels
(overfit discipline: the sample must not depend on what the answer is). Allocation is proportional to stratum
size with the largest-remainder method; within a stratum, ids are sampled with random.Random(seed) over the
sorted id list, so the same (n, seed) always yields the same pilot.

  python -m compilescholar.eval.sc.sample --bench-dir <shadow> --n 50 --seed 42 --out pilot50_ids.txt
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from . import protocol


def stratified_sample(bench_dir: Path, n: int = 50, seed: int = 42) -> tuple[list[str], dict]:
    strata: dict[tuple[str, str], list[str]] = {}
    for qt in protocol.QUERY_TYPES:
        for q in protocol.load_queries(bench_dir, qt):
            strata.setdefault((q["type"], q.get("paper_domain") or "?"), []).append(q["id"])
    total = sum(len(v) for v in strata.values())
    n = min(n, total)

    # largest-remainder proportional allocation; ties broken by stratum size, then key (deterministic)
    quotas = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    rem = n - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), -len(strata[k]), k)):
        if rem <= 0:
            break
        alloc[k] += 1
        rem -= 1

    rng = random.Random(seed)
    picked: list[str] = []
    for k in sorted(strata):                            # deterministic emit order
        ids = sorted(strata[k])
        take = min(alloc.get(k, 0), len(ids))
        if take:
            picked.extend(rng.sample(ids, take))
    report = {"n": len(picked), "seed": seed, "total_queries": total,
              "allocation": {f"{qt}|{dom}": {"pool": len(strata[(qt, dom)]), "picked": alloc.get((qt, dom), 0)}
                             for (qt, dom) in sorted(strata)}}
    return picked, report


def _type_index(bench_dir: Path) -> dict[str, str]:
    out = {}
    for qt in protocol.QUERY_TYPES:
        for q in protocol.load_queries(bench_dir, qt):
            out[q["id"]] = q["type"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    picked, report = stratified_sample(a.bench_dir, a.n, a.seed)
    tix = _type_index(a.bench_dir)
    report["by_type"] = dict(Counter(tix[q] for q in picked))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(sorted(picked)) + "\n", encoding="utf-8")
    (a.out.with_suffix(".report.json")).write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                   encoding="utf-8")
    print(f"wrote {len(picked)} ids -> {a.out}")
    print(json.dumps({"by_type": report["by_type"],
                      "n_strata_with_picks": sum(1 for v in report["allocation"].values() if v["picked"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
