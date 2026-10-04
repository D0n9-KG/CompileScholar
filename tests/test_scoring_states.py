# -*- coding: utf-8 -*-
"""W1-1: a question is answered, missing (no row — the system gave no answer; scores 0), or a judge error (a row
with error/_errors or without the four facets — never scored; summarize() refuses unless explicitly allowed).
A score file that does not exist raises instead of silently scoring every question 0."""
from __future__ import annotations

import json

import pytest

from compilescholar.eval.cs2 import scoring as S

GOOD = {"ingredient_recall": {"ingredient_recall": 0.5}, "answer_precision": {"answer_precision": 1.0},
        "citation": {"citation_recall": 0.25, "citation_precision": 0.75}}


def _write(tmp_path, d):
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_states():
    assert S.state(GOOD) == "answered"
    assert S.state(None) == "missing"
    assert S.state({"_errors": ["TimeoutError"]}) == "judge_error"
    assert S.state({"error": "x"}) == "judge_error"
    assert S.state({"ingredient_recall": {"ingredient_recall": 0.5}}) == "judge_error"


def test_missing_scores_zero(tmp_path):
    f = _write(tmp_path, {"q1": GOOD})
    s = S.summarize(f, ["q1", "q2"])
    assert s["n_missing_as_zero"] == 1 and s["n_judge_error"] == 0
    assert s["mean"]["global"] == pytest.approx(0.625 / 2)


def test_judge_error_refused(tmp_path):
    f = _write(tmp_path, {"q1": GOOD, "q2": {"_errors": ["boom"]}})
    with pytest.raises(S.JudgeErrorRows) as e:
        S.summarize(f, ["q1", "q2"])
    assert "q2" in str(e.value)
    s = S.summarize(f, ["q1", "q2"], allow_judge_errors=True)
    assert s["n_judge_error"] == 1 and s["judge_error"] == ["q2"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.summarize(str(tmp_path / "nope.json"), ["q1"])


def test_v9b_test_files_unchanged():
    """The frozen v9b test score files have no judge-error rows, so the new rule leaves every number unchanged."""
    import _characterize_impl as C
    facets = json.load(open(C.FIX / "cs2_test_facets.json", encoding="utf-8"))
    qids = json.load(open(C.FIX / "cs2_test_qids.json", encoding="utf-8"))
    gold = json.load(open(C.FIX / "goldens_scoring.json", encoding="utf-8"))["means"]
    import tempfile, os
    d = tempfile.mkdtemp()
    for name, rows in facets.items():
        p = os.path.join(d, name + ".json")
        json.dump(rows, open(p, "w"))
        s = S.summarize(p, qids)
        assert s["n_judge_error"] == 0
        assert round(s["mean"]["global"], 12) == gold[name]["global"]
