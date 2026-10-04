# -*- coding: utf-8 -*-
"""W1-10: when the LLM returns nothing usable, the answer still completes and the trace counts each fallback."""
from __future__ import annotations

import _characterize_impl as C
import compilescholar.answer.pipeline as AP
from compilescholar.kb.store import KB


def test_degradation_counters(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_CUTOFF", "2025-05")
    monkeypatch.setattr(AP, "chat", lambda prompt, max_tokens=6000, temperature=0.2: "")  # dead LLM
    monkeypatch.setattr(AP, "_sciverse", lambda: C.FakeSciverse())
    kb = KB(str(C.kb_fixture_dir()), embed=C.fake_embed)
    r = AP.answer("How do graph neural networks handle molecules?", kb, use_cite=False, word_budget=1000)
    d = r["trace"]["degradation"]
    assert d["plan_unparseable"] == 2 and d["plan_fallback"] == 1     # both temperatures failed -> 1-section fallback
    assert d["screen_unparseable"] >= 1 and d["write_empty"] == 1      # one section, written empty
    assert r["sections"] == [] or all(not s["text"] for s in r["sections"])


def test_healthy_run_counts_zero(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_CUTOFF", "2025-05")
    monkeypatch.setattr(AP, "chat", C.fake_chat)
    monkeypatch.setattr(AP, "_sciverse", lambda: C.FakeSciverse())
    kb = KB(str(C.kb_fixture_dir()), embed=C.fake_embed)
    r = AP.answer("How do graph neural networks handle molecular property prediction?", kb, use_cite=False)
    assert all(v == 0 for k, v in r["trace"]["degradation"].items() if k != "cite_section_empty")
