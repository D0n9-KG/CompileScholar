# -*- coding: utf-8 -*-
"""W1-6 variants change only snippets; trimming keeps the sentences that match the citing context."""
from __future__ import annotations

import json

from compilescholar.eval.cs2 import format_align as F

ROWS = [{"qid": "q", "question": "?", "sections": [{"title": "T", "text": "Graph networks predict molecular energy [1]. "
         "Other work studies images [2].", "citations": [
            {"id": "[1]", "title": "GNN paper", "snippets": [
                "We introduce a new benchmark. Graph neural networks predict molecular energy accurately. "
                "Training uses 8 GPUs. The code is released."]},
            {"id": "[2]", "title": "Vision", "snippets": ["Images are studied here in depth. " * 20]}]}]}]


def test_trim_keeps_relevant_sentence_and_cap():
    v = F.variant(ROWS, "trim", cap=240)
    s1 = v[0]["sections"][0]["citations"][0]["snippets"][0]
    assert "Graph neural networks predict molecular energy" in s1 and len(s1) <= 240
    s2 = v[0]["sections"][0]["citations"][1]["snippets"][0]
    assert len(s2) <= 240


def test_only_snippets_change():
    for kind in ("trim", "title"):
        v = F.variant(ROWS, kind)
        a, b = json.loads(json.dumps(ROWS)), v
        for sa, sb in zip(a[0]["sections"], b[0]["sections"]):
            assert sa["text"] == sb["text"] and sa["title"] == sb["title"]
            assert [c["id"] for c in sa["citations"]] == [c["id"] for c in sb["citations"]]
            assert [c["title"] for c in sa["citations"]] == [c["title"] for c in sb["citations"]]
    assert all(not c["snippets"] for c in F.variant(ROWS, "title")[0]["sections"][0]["citations"])
    assert ROWS[0]["sections"][0]["citations"][0]["snippets"][0].startswith("We introduce")  # input untouched


def test_expand_uses_abstract_by_title():
    v = F.variant(ROWS, "expand", abstracts={"gnn paper": "FULL ABSTRACT"})
    assert v[0]["sections"][0]["citations"][0]["snippets"] == ["FULL ABSTRACT"]
    assert v[0]["sections"][0]["citations"][1]["snippets"] == ROWS[0]["sections"][0]["citations"][1]["snippets"]
