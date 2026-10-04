# -*- coding: utf-8 -*-
"""Scoring goldens from the OLD cs2_scoring / paired_stats on the frozen v9b CS2 test facets (run once, before the move).
The old modules now live in legacy/; regenerating requires a checkout of tag cs2-test-v9b-final.

Usage:  python tests/fixtures/characterize/make_goldens_scoring.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _characterize_impl as C  # noqa: E402

OLD_CS2 = C.REPO / ".research_tmp" / "experiments" / "benchmarks" / "cs2"


def load_scoring(impl: str):
    if impl == "old":
        if str(OLD_CS2) not in sys.path:
            sys.path.insert(0, str(OLD_CS2))
        import cs2_scoring
        import paired_stats
        return cs2_scoring, paired_stats
    if impl == "new":
        from compilescholar.eval import stats
        from compilescholar.eval.cs2 import scoring
        return scoring, stats
    raise ValueError(impl)


def compute(impl: str) -> dict:
    S, P = load_scoring(impl)
    facets = json.load(open(C.FIX / "cs2_test_facets.json", encoding="utf-8"))
    qids = json.load(open(C.FIX / "cs2_test_qids.json", encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp())
    files = {}
    for name, d in facets.items():
        p = tmp / f"{name}.json"
        json.dump(d, open(p, "w"))
        files[name] = str(p)
    out = {"means": {}, "paired": {}}
    for name, f in files.items():
        s = S.summarize(f, qids)
        out["means"][name] = {k: round(v, 12) for k, v in s["mean"].items()}
        out["means"][name]["n_missing_as_zero"] = s["n_missing_as_zero"]
    ref = P.per_q([files["ours_r1"], files["ours_r2"]], qids)
    for name in ("harness", "gptr", "elicit", "scispace", "openai_dr"):
        b = P.per_q([files[name]], qids)
        row = {}
        for k in P.KEYS:
            d = [ref[q][k] - b[q][k] for q in qids]
            lo, hi = P.boot_ci(d)
            row[k] = [round(sum(d) / len(d), 12), round(lo, 12), round(hi, 12), round(P.perm_p(d), 12)]
        out["paired"][name] = row
    out["holm"] = P.holm([0.01, 0.04, 0.03, 0.2, 0.001])
    # missing-answer handling: drop 3 questions from one arm -> they count as 0
    d = dict(facets["harness"])
    for q in qids[:3]:
        d.pop(q, None)
    p = tmp / "harness_missing3.json"
    json.dump(d, open(p, "w"))
    s = S.summarize(str(p), qids)
    out["missing3"] = {"n_missing_as_zero": s["n_missing_as_zero"], "global": round(s["mean"]["global"], 12)}
    # a judge-error row (scorer structure missing) was counted as 0 by the OLD code. W1-1 changed this on purpose:
    # the new code refuses to summarize unless allow_judge_errors=True, and then still scores it 0 — so the number
    # with the flag set must equal the old behaviour.
    d = dict(facets["harness"])
    d[qids[0]] = {"_errors": ["TimeoutError: x"]}
    p = tmp / "harness_err1.json"
    json.dump(d, open(p, "w"))
    try:
        s = S.summarize(str(p), qids, allow_judge_errors=True)
        n0 = s["n_missing_as_zero"] + s.get("n_judge_error", 0)
    except TypeError:  # old signature
        s = S.summarize(str(p), qids)
        n0 = s["n_missing_as_zero"]
    out["judge_error1_old_behaviour"] = {"n_missing_as_zero": n0, "global": round(s["mean"]["global"], 12)}
    return out


if __name__ == "__main__":
    g = compute("old")
    json.dump(g, open(C.FIX / "goldens_scoring.json", "w", encoding="utf-8"), indent=1, sort_keys=True)
    print({k: round(v["global"], 3) for k, v in g["means"].items()})
