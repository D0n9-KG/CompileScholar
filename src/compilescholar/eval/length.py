# -*- coding: utf-8 -*-
"""W1-7: length as a covariate. score_q,s = a_q + b_s + c * log(words_q,s) + e  (question fixed effects a_q,
system effects b_s, common length slope c), estimated by OLS after within-question demeaning; question-cluster
bootstrap CIs. Reported next to the main table, never used to adjust it (PREREG §4)."""
from __future__ import annotations

import math
import random

import numpy as np


def words(sections: list[dict]) -> int:
    return sum(len((s.get("text") or "").split()) for s in sections or [])


def fit(score: dict[str, dict[str, float]], length: dict[str, dict[str, int]], ref: str) -> dict:
    """score[system][qid], length[system][qid]; ref = baseline system (its effect fixed at 0).
    Returns {"c": length slope, "b": {system: effect vs ref}} on demeaned data."""
    systems = sorted(score)
    qids = sorted(set.intersection(*[set(score[s]) for s in systems]))
    others = [s for s in systems if s != ref]
    X, y, groups = [], [], []
    for q in qids:
        rows = [(s, score[s][q], math.log(max(1, length[s][q]))) for s in systems]
        ym = sum(r[1] for r in rows) / len(rows)
        lm = sum(r[2] for r in rows) / len(rows)
        dm = {o: sum(1.0 for r in rows if r[0] == o) / len(rows) for o in others}
        for s, v, lw in rows:
            X.append([lw - lm] + [(1.0 if s == o else 0.0) - dm[o] for o in others])
            y.append(v - ym)
            groups.append(q)
    X, y = np.asarray(X), np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"c": float(beta[0]), "b": {o: float(beta[i + 1]) for i, o in enumerate(others)}, "n_questions": len(qids),
            "_X": X, "_y": y, "_groups": groups, "_others": others}


def fit_with_ci(score, length, ref, B: int = 2000, seed: int = 0) -> dict:
    base = fit(score, length, ref)
    qids = sorted(set(base["_groups"]))
    idx = {q: [i for i, g in enumerate(base["_groups"]) if g == q] for q in qids}
    X, y = base["_X"], base["_y"]
    rng = random.Random(seed)
    cs, bs = [], {o: [] for o in base["_others"]}
    for _ in range(B):
        rows = [i for _ in qids for i in idx[qids[rng.randrange(len(qids))]]]
        beta, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
        cs.append(beta[0])
        for i, o in enumerate(base["_others"]):
            bs[o].append(beta[i + 1])

    def ci(v):
        v = sorted(v)
        return float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])
    return {"c": base["c"], "c_ci": ci(cs), "b": {o: (base["b"][o], *ci(bs[o])) for o in base["_others"]},
            "n_questions": base["n_questions"]}
