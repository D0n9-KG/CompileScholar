# -*- coding: utf-8 -*-
"""BCa interval: brackets the mean, close to the percentile interval for a symmetric sample, and shifts the way the
skew points for a skewed one; deterministic per seed."""
from __future__ import annotations

import random

from compilescholar.eval import stats as S


def test_bca_symmetric_close_to_percentile():
    rng = random.Random(1)
    d = [rng.gauss(0.05, 0.1) for _ in range(100)]
    m = sum(d) / len(d)
    lo, hi = S.bca_ci(d)
    plo, phi = S.boot_ci(d)
    assert lo < m < hi
    assert abs(lo - plo) < 0.01 and abs(hi - phi) < 0.01
    assert S.bca_ci(d) == (lo, hi)


def test_bca_skewed_shifts_right():
    d = [0.0] * 15 + [0.9, 1.0, 0.8]
    lo, hi = S.bca_ci(d, B=4000)
    plo, phi = S.boot_ci(d, B=4000)
    m = sum(d) / len(d)
    assert lo < m < hi
    assert hi >= phi  # right-skewed sample: BCa extends the upper end


def test_constant_sample():
    lo, hi = S.bca_ci([0.1] * 10, B=500)
    assert abs(lo - 0.1) < 1e-12 and abs(hi - 0.1) < 1e-12
