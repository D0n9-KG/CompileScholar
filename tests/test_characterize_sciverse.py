# -*- coding: utf-8 -*-
"""Old retrieval.sources.SciverseClient.semantic_search vs compilescholar.sources.sciverse: identical request payloads
(including the server-side cutoff filter) and identical candidates, for the same fake server responses."""
from __future__ import annotations

import os

import pytest

HITS = [
    {"doc_id": "d1", "chunk_id": "c1", "chunk": "x", "title": "Paper One", "publication_published_year": 2021,
     "abstract": "Abstract one " * 10, "score": 0.9, "citation_count": 3, "primary_topic": "t"},
    {"doc_id": "d1", "chunk_id": "c2", "chunk": "y", "title": "Paper One", "publication_published_year": 2021,
     "abstract": "dup chunk", "score": 0.8},
    {"doc_id": "d2", "title": "Paper Two", "publication_published_year": "2019", "abstract": "Two " * 30, "score": 1},
    {"doc_id": "", "title": None, "publication_published_year": None, "score": "bad"},
    "not-a-dict",
    {"doc_id": "d3", "title": "Paper Three", "publication_venue_name_unified": "NeurIPS", "score": 0.1},
]


def _run(client_cls, cutoff, response, kwargs_name):
    sent = []

    def fake(method, path, *, payload=None, query=None, timeout_seconds=30):
        sent.append((method, path, payload, query, timeout_seconds))
        if isinstance(response, Exception):
            raise response
        return response
    if cutoff is None:
        os.environ.pop("KNOWLEDGE_CUTOFF", None)
    else:
        os.environ["KNOWLEDGE_CUTOFF"] = cutoff
    c = client_cls(token="tok", **{kwargs_name: fake}, timeout_seconds=60)
    out = c.semantic_search("graph neural networks", limit=3)
    return sent, [(x.source_name, x.query_kind, x.status, x.source_record_id, x.title, x.year, x.venue,
                   x.candidate_score, x.raw, x.error_summary is not None) for x in out]


@pytest.mark.parametrize("cutoff", ["2025-05", "2023-06", None])
@pytest.mark.parametrize("response", ["hits", "notdict", "error"])
def test_semantic_search_equivalent(cutoff, response):
    import retrieval.sources as old
    import compilescholar.sources.sciverse as new
    resp = {"hits": HITS} if response == "hits" else ([] if response == "notdict" else old.SourceAdapterError("boom"))
    resp_new = resp if response != "error" else new.SourceAdapterError("boom")
    a = _run(old.SciverseClient, cutoff, resp, "request_json")
    b = _run(new.SciverseClient, cutoff, resp_new, "request_json_fn")
    assert a[0] == b[0]
    assert a[1] == b[1]


def test_blank_query_and_no_token():
    import compilescholar.sources.sciverse as new
    assert new.SciverseClient(token="t").semantic_search("  ") == []
    r = new.SciverseClient(token="").semantic_search("q")
    assert len(r) == 1 and r[0].status == "blocked"
