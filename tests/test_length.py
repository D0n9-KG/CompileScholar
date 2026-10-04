# -*- coding: utf-8 -*-
"""Length regression recovers a known slope and known system effects on synthetic data."""
from __future__ import annotations

import math
import random

from compilescholar.eval import length as L


def test_recovers_slope_and_effects():
    rng = random.Random(0)
    score, ln = {"a": {}, "b": {}, "c": {}}, {"a": {}, "b": {}, "c": {}}
    eff = {"a": 0.0, "b": 0.10, "c": -0.05}
    for i in range(80):
        q = f"q{i}"
        base = rng.uniform(0.2, 0.8)
        for s in score:
            w = rng.randint(300, 3000)
            ln[s][q] = w
            score[s][q] = base + eff[s] + 0.04 * math.log(w) + rng.gauss(0, 0.01)
    r = L.fit_with_ci(score, ln, ref="a", B=300)
    assert abs(r["c"] - 0.04) < 0.01
    assert abs(r["b"]["b"][0] - 0.10) < 0.01 and abs(r["b"]["c"][0] + 0.05) < 0.01
    lo, hi = r["c_ci"]
    assert lo < 0.04 < hi


def test_words():
    assert L.words([{"text": "a b c"}, {"text": "d"}]) == 4 and L.words([]) == 0
