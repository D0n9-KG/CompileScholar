# -*- coding: utf-8 -*-
"""W1-13: the same paper reached through different channels (KB, external search, citation expansion) gets one
citation number; different papers never merge."""
from __future__ import annotations

from compilescholar.answer import pipeline as AP


def _e(eid, src, key, title, snippet, arxiv=None):
    return {"eid": eid, "src": src, "paper_key": key, "title": title, "snippet": snippet, "year": 2021, "arxiv": arxiv,
            "sections": {0}}


def test_same_title_across_channels_merges():
    ev = [_e("E1", "kb", "kb:p1", "Attention Is All You Need", "kb excerpt"),
          _e("E2", "ext", "ext:abc", "Attention is all you need.", "ext excerpt"),
          _e("E3", "ext", "ext:def", "BERT: Pre-training of Deep Bidirectional Transformers", "bert")]
    secs, st = AP.assemble([{"title": "S"}], ["A [E1]. B [E2]. C [E3]."], ev)
    c = secs[0]["citations"]
    assert [x["id"] for x in c] == ["[1]", "[2]"]
    assert c[0]["snippets"] == ["kb excerpt", "ext excerpt"]
    assert secs[0]["text"] == "A [1]. B [1]. C [2]."
    assert st["merged_identities"] == 1


def test_arxiv_id_merges_even_if_titles_differ():
    ev = [_e("E1", "kb", "kb:p1", "Old title v1", "a", arxiv="1706.03762"),
          _e("E2", "cite", "cite:x", "Final title", "b", arxiv="1706.03762v5")]
    secs, _ = AP.assemble([{"title": "S"}], ["X [E1][E2]."], ev)
    assert len(secs[0]["citations"]) == 1 and secs[0]["text"] == "X [1]."


def test_short_or_generic_titles_do_not_merge():
    ev = [_e("E1", "kb", "kb:p1", "Introduction", "a"), _e("E2", "ext", "ext:q", "Introduction", "b")]
    secs, _ = AP.assemble([{"title": "S"}], ["X [E1]. Y [E2]."], ev)
    assert len(secs[0]["citations"]) == 2
