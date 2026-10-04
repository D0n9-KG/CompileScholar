# -*- coding: utf-8 -*-
"""Characterization tests (upgrade W6 S2): every implementation in IMPLS must reproduce the goldens generated from the
OLD answer path byte for byte. Goldens: tests/fixtures/characterize/goldens*.json (made by make_goldens*.py, never
regenerated from new code). No network, no LLM."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import _characterize_impl as C

FIXDIR = C.FIX


def _run_compute(script: str, impl: str) -> dict:
    """Run a golden generator's compute(impl) in a fresh interpreter (module-level state such as monkeypatched
    retrievers, env-dependent semaphores and sys.path must not leak between implementations or into other tests)."""
    code = (f"import sys, json; sys.path.insert(0, {str(C.REPO / 'tests')!r}); sys.path.insert(0, {str(FIXDIR)!r}); "
            f"import {script} as M; print(json.dumps(M.compute({impl!r}), ensure_ascii=False, sort_keys=True))")
    env = {**os.environ, "PYTHONUTF8": "1", "LLM_PROVIDER_ALLOWLIST": "none", "KNOWLEDGE_CUTOFF": "2025-05"}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", env=env,
                       timeout=900, cwd=str(C.REPO))
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module", params=C.IMPLS)
def answer_path(request):
    return _run_compute("make_goldens", request.param)


@pytest.fixture(scope="module", params=C.IMPLS)
def scoring(request):
    return _run_compute("make_goldens_scoring", request.param)


GOLD = json.load(open(FIXDIR / "goldens.json", encoding="utf-8"))
GOLD_SCORING = json.load(open(FIXDIR / "goldens_scoring.json", encoding="utf-8"))


@pytest.mark.parametrize("key", sorted(GOLD))
def test_answer_path_matches_golden(answer_path, key):
    assert C.canon(answer_path[key]) == C.canon(GOLD[key])


@pytest.mark.parametrize("key", sorted(GOLD_SCORING))
def test_scoring_matches_golden(scoring, key):
    assert C.canon(scoring[key]) == C.canon(GOLD_SCORING[key])


def test_scoring_reproduces_reported_test_table():
    """The frozen v9b CS2 test numbers reported in PAPER-DRAFT §6.2 (ours = mean of two runs)."""
    m = GOLD_SCORING["means"]
    ours = (m["ours_r1"]["global"] + m["ours_r2"]["global"]) / 2
    assert round(ours, 3) == 0.829
    assert {k: round(m[k]["global"], 3) for k in ("scispace", "elicit", "harness", "gptr", "openai_dr")} == \
        {"scispace": 0.837, "elicit": 0.798, "harness": 0.744, "gptr": 0.736, "openai_dr": 0.684}
    assert [round(x, 3) for x in GOLD_SCORING["paired"]["harness"]["global"][:3]] == [0.086, 0.066, 0.107]
