# -*- coding: utf-8 -*-
"""Old _shared/mcp/{cc_compat_proxy,retrieval_mcp}.py vs compilescholar.baselines.harness.{proxy,mcp_server}:
the proxy's request/response rewrites and the MCP server's open-mode search output must be identical."""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from _characterize_impl import OLD_TOOLS, REPO

OLD_MCP = REPO / ".research_tmp" / "experiments" / "benchmarks" / "_shared" / "mcp"


def _load_old(name: str):
    if str(OLD_TOOLS) not in sys.path:
        sys.path.insert(0, str(OLD_TOOLS))
    spec = importlib.util.spec_from_file_location(f"old_{name}", OLD_MCP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REQS = [
    {"model": "m", "system": "S", "messages": [{"role": "user", "content": "hi"},
                                              {"role": "system", "content": [{"type": "text", "text": "late sys"}]}]},
    {"model": "m", "system": [{"type": "text", "text": "S"}], "messages": [{"role": "system", "content": "x"},
                                                                            {"role": "user", "content": "q"}]},
    {"model": "m", "messages": [{"role": "user", "content": "no system"}]},
]
RESPS = [
    {"content": [{"type": "tool_use", "id": "t", "name": "list_corpus"}, {"type": "text", "text": "a"}]},
    {"content": [{"type": "thinking", "text": "hmm"}, {"type": "tool_use", "id": "t", "name": "x", "input": {"q": 1}}]},
    {"error": "x"},
]
SSE = [b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"n"}}',
       b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","text":"abc"}}',
       b'data: {"type":"message_stop"}', b"event: ping", b"data: not json"]


def test_proxy_rewrites_identical():
    old = _load_old("cc_compat_proxy")
    import compilescholar.baselines.harness.proxy as new
    for r in REQS:
        b = json.dumps(r).encode()
        assert old.fix(b) == new.fix(b)
    assert old.fix(b"not json") == new.fix(b"not json")
    for r in RESPS:
        b = json.dumps(r).encode()
        assert old.norm_json(b) == new.norm_json(b)
    for ln in SSE:
        assert old.norm_sse_line(ln) == new.norm_sse_line(ln)


class _Cand:
    def __init__(self, status, title, year, raw, venue=None, score=0.5):
        self.status, self.title, self.year, self.raw, self.venue, self.candidate_score = status, title, year, raw, venue, score


class _FakeClient:
    def semantic_search(self, q, limit=10):
        return [_Cand("ready", f"T{i}", 2019 + i, {"doc_id": f"d{i}", "chunk": "c" * 500, "chunk_id": f"c{i}"}, "V")
                for i in range(8)] + [_Cand("failed", None, None, {})]

    def read_content(self, *, doc_id, chunk_id=None, offset=None, limit=None):
        return {"text": f"{doc_id}:{chunk_id}:{offset}:{limit} " + "x" * 13000}


@pytest.mark.parametrize("cutoff", ["2025-05", "2023-06", None])
@pytest.mark.parametrize("before_year", [None, 2022])
def test_mcp_open_search_identical(cutoff, before_year, monkeypatch):
    if cutoff is None:
        monkeypatch.delenv("KNOWLEDGE_CUTOFF", raising=False)
    else:
        monkeypatch.setenv("KNOWLEDGE_CUTOFF", cutoff)
    monkeypatch.setenv("RETRIEVAL_MCP_MODE", "open")
    old = _load_old("retrieval_mcp")
    import compilescholar.baselines.harness.mcp_server as new
    monkeypatch.setattr(old, "_sciverse", lambda: _FakeClient())
    monkeypatch.setattr(new, "_sciverse", lambda: _FakeClient())
    assert old.search_papers("q", k=5, before_year=before_year) == new.search_papers("q", k=5, before_year=before_year)
    assert old.fetch_chunk("d1", chunk_id="c1") == new.fetch_chunk("d1", chunk_id="c1")
    assert old.fetch_chunk("d2", offset=3, limit=2) == new.fetch_chunk("d2", offset=3, limit=2)
    assert old.list_corpus() == new.list_corpus()
