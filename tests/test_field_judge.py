# -*- coding: utf-8 -*-
"""W1-4 field-test judge: all candidates judged (chunked, global ids), parse failures raise, small inputs send the
same prompt as before."""
from __future__ import annotations

import json
import re

import pytest

import compilescholar.eval.field as F


def _fake(match_fn, sent):
    def call(prompt, **kw):
        sent.append(prompt)
        gold = re.findall(r"^(G\d+):", prompt, re.M)
        cand = re.findall(r"^(C\d+):", prompt, re.M)
        return match_fn(gold, cand)
    return call


def test_candidates_beyond_250_can_match(monkeypatch):
    sent = []
    # every gold matches only the last candidate
    monkeypatch.setattr(F, "call_paratera", _fake(lambda g, c: json.dumps({"matches": {x: ([c[-1]] if "C600" in c else [])
                                                                                       for x in g}}), sent))
    m = F.judge_match(["g1", "g2"], [f"cand {i}" for i in range(600)], "x")
    assert m == {"G1": ["C600"], "G2": ["C600"]}
    assert len(sent) == 3  # 600 candidates / 250 per call


def test_ids_outside_chunk_are_ignored(monkeypatch):
    sent = []
    monkeypatch.setattr(F, "call_paratera", _fake(lambda g, c: json.dumps({"matches": {x: ["C1", "C999"] for x in g}}), sent))
    m = F.judge_match(["g"], [f"c{i}" for i in range(300)], "x")
    assert m == {"G1": ["C1"]}  # C1 valid only in chunk 1; C999 never in range


def test_parse_failure_raises(monkeypatch):
    monkeypatch.setattr(F, "call_paratera", lambda prompt, **kw: "not json")
    with pytest.raises(F.JudgeFailed):
        F.judge_match(["g"], ["c"], "x")


def test_small_input_prompt_unchanged(monkeypatch):
    sent = []
    monkeypatch.setattr(F, "call_paratera", _fake(lambda g, c: json.dumps({"matches": {}}), sent))
    F.judge_match(["gold a", "gold b"], ["cand a", "cand b", "cand c"], "limitation")
    old = F.MATCH.format(what="limitation", gold="G1: gold a\nG2: gold b", cand="C1: cand a\nC2: cand b\nC3: cand c")
    assert sent == [old]
