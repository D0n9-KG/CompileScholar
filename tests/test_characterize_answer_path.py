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


@pytest.fixture(scope="module", params=C.SCORING_IMPLS)
def scoring(request):
    return _run_compute("make_goldens_scoring", request.param)


GOLD = json.load(open(FIXDIR / "goldens.json", encoding="utf-8"))
GOLD_SCORING = json.load(open(FIXDIR / "goldens_scoring.json", encoding="utf-8"))


INTENDED = json.load(open(FIXDIR / "goldens_intended.json", encoding="utf-8"))
ADDED_DEGRADATION_KEYS = ("plan_unparseable", "plan_fallback", "screen_unparseable", "screen_kept_all", "write_empty",
                          "ext_query_failed", "cite_section_empty")


@pytest.mark.parametrize("key", sorted(GOLD))
def test_answer_path_matches_golden(answer_path, key):
    """Pre-move goldens, except entries changed on purpose (tests/fixtures/characterize/intended_changes.py)."""
    if key == "answer":
        for name, v in GOLD["answer"].items():
            want = INTENDED.get(f"answer.{name}", v)
            got = json.loads(C.canon(answer_path["answer"][name]))
            # W1-10 adds trace.degradation; W1-13 adds trace.assemble.merged_identities (new fields only)
            assert set(got["result"]["trace"].pop("degradation")) == set(ADDED_DEGRADATION_KEYS)
            assert isinstance(got["result"]["trace"]["assemble"].pop("merged_identities"), int)
            assert C.canon(got) == C.canon(want), name
        return
    assert C.canon(answer_path[key]) == C.canon(GOLD[key])


def test_intended_change_w1_12_only_drops_post_cutoff_kb_evidence():
    old = GOLD["answer"]["task_context_cutoff"]["result"]["trace"]["evidence"]
    new = INTENDED["answer.task_context_cutoff"]["result"]["trace"]["evidence"]
    assert all((e.get("year") or 0) <= 2023 for e in new if e["src"] == "kb")
    assert any(e["src"] == "kb" and (e.get("year") or 0) >= 2024 for e in old)
    # external evidence is unaffected by the change (ids may shift because KB rows are fewer)
    strip = lambda rows: sorted((e["src"], e["paper_key"], e["snippet"]) for e in rows if e["src"] != "kb")  # noqa: E731
    assert strip(old) == strip(new)


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
