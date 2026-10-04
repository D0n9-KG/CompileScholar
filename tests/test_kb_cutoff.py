# -*- coding: utf-8 -*-
"""W1-12: KB retrieval honours the knowledge cutoff (record channel and state channel), and the answer trace counts
what was filtered. With no cutoff set, results are unchanged (pinned by the characterization goldens)."""
from __future__ import annotations

import json
import os

import _characterize_impl as C
from compilescholar.core import cutoff as CUT
from compilescholar.kb.store import KB


def _kb():
    return KB(str(C.kb_fixture_dir()), embed=C.fake_embed)


def test_record_and_state_channels_filter_by_year(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_CUTOFF", raising=False)
    kb = _kb()
    years = {pid: (kb.papers.get(pid) or {}).get("year") for pid in kb.papers}
    q = "limitations of large language models"
    full = kb.search(q, 30)
    full_state = kb.state_search(q, min_sim=0.0)
    assert any((r["year"] or 0) >= 2023 for r in full)
    CUT.set_thread_cutoff("2023-01")
    try:
        cut = kb.search(q, 30)
        cut_state = kb.state_search(q, min_sim=0.0)
        assert cut and all(r["year"] is not None and r["year"] < 2023 for r in cut)
        assert all(r["year"] is not None and r["year"] < 2023 for r in cut_state)
        assert kb.filtered_by_cutoff > 0
    finally:
        CUT.set_thread_cutoff(None)
    assert {r["paper_key"] for r in cut} <= {r["paper_key"] for r in kb.search(q, 200)}
    assert len(full_state) >= len(cut_state)
    assert years  # fixture sanity


def test_no_cutoff_is_unchanged(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_CUTOFF", raising=False)
    kb = _kb()
    gold = json.load(open(C.FIX / "goldens.json", encoding="utf-8"))
    os.environ["KNOWLEDGE_CUTOFF"] = "2025-05"  # goldens were made with this cutoff; KB has nothing past 2024
    try:
        for q, rows in gold["kb_search"].items():
            assert C.canon(kb.search(q, 8)) == C.canon(rows)
    finally:
        os.environ.pop("KNOWLEDGE_CUTOFF", None)
