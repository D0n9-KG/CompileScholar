# -*- coding: utf-8 -*-
"""SC eval adapter, offline (no DB, no network): SystemSearch.state_expand channel filtering
(pool-only mapping, withheld, as-of month guard, degradation recording) and the agent's RRF fuse step.
Fakes mirror the tools.api contract: family_neighbors(seeds, as_of, k) -> {"local": [{"id": ...}, ...]}."""
from __future__ import annotations

from compilescholar.eval.sc.agent import AgentConfig, SCSearchAgent
from compilescholar.eval.sc.backends import SystemSearch


class _IdMap:
    def __init__(self, pairs):
        self.d2p = dict(pairs)

    def paper_of(self, d):
        return self.d2p.get(d)

    def docs_of(self, p):
        return [d for d, x in self.d2p.items() if x == p]


class _Corpus:
    titles: dict = {}

    def __init__(self, dates):
        self.dates = dates

    def snippet(self, d, n):
        return ""


class _Tools:
    def __init__(self, rows=None, boom=None):
        self.rows = rows or []
        self.boom = boom
        self.calls = []

    def family_neighbors(self, seeds, as_of, k):
        if self.boom:
            raise self.boom
        self.calls.append((list(seeds), as_of, k))
        return {"local": self.rows}


def test_state_expand_filters_pool_withheld_and_date():
    tools = _Tools(rows=[{"id": "p2"}, {"id": "pX"}, {"id": "p3"}, {"id": "p1"}])
    be = SystemSearch(_IdMap([("d1", "p1"), ("d2", "p2"), ("d3", "p3")]),
                      _Corpus({"d1": "2020-01-15", "d2": "2021-03-01", "d3": "2025-09-30"}), tools=tools)
    got = be.state_expand(["d1"], "2025-06-30", 10, withheld={"d2"})
    # pX is not pool; d2 withheld; d3 published 2025-09 > as-of 2025-06; d1 passes (the agent dedups on add)
    assert got == ["d1"]
    assert be.degraded == []
    assert tools.calls == [(["p1"], "2025-06-30", 30)]     # seeds mapped, k overfetched by POOL_FACTOR


def test_state_expand_same_month_passes():
    tools = _Tools(rows=[{"id": "p3"}])
    be = SystemSearch(_IdMap([("d1", "p1"), ("d3", "p3")]),
                      _Corpus({"d1": "2020-01-15", "d3": "2025-06-02"}), tools=tools)
    assert be.state_expand(["d1"], "2025-06-30", 10, withheld=set()) == ["d3"]   # protocol: same month legal


def test_state_expand_records_degradation():
    be = SystemSearch(_IdMap([("d1", "p1")]), _Corpus({"d1": "2020-01-15"}),
                      tools=_Tools(boom=RuntimeError("cognition store missing")))
    assert be.state_expand(["d1"], "2025-06-30", 5, withheld=set()) == []
    assert be.degraded and "family_neighbors" in be.degraded[0]


def test_state_expand_no_mapped_seeds():
    tools = _Tools(rows=[{"id": "p2"}])
    be = SystemSearch(_IdMap([("d1", "p1")]), _Corpus({"d1": "2020-01-15"}), tools=tools)
    assert be.state_expand(["dX"], "2025-06-30", 5, withheld=set()) == []
    assert tools.calls == []


def test_rrf_fusion_keeps_llm_head_and_rescues_discovery_confident_doc():
    a = SCSearchAgent(backend=None, cfg=AgentConfig())
    ranked = ["x1", "x2", "x3", "p"]          # the LLM ranker demoted positive p to last
    disc = ["p", "x9", "x1", "x2", "x3"]      # retrieval discovered p first; x9 never reached the LLM list
    fused = a._rrf(ranked, disc)
    assert fused[0] == "x1" and fused[1] == "p"          # strong in one list + present in the other beats
    assert fused[-1] == "x9"                              # single-list-only docs sink to the tail
    assert set(fused) == set(ranked) | set(disc)


def test_rrf_identical_lists_are_identity():
    a = SCSearchAgent(backend=None, cfg=AgentConfig())
    ranked = ["a", "b", "c"]
    assert a._rrf(ranked, ranked) == ranked


def test_pool_cap_fits_llm_budget():
    cfg = AgentConfig()
    judge_calls = -(-cfg.pool_cap // cfg.judge_batch)       # ceil
    assert 1 + judge_calls + 1 <= cfg.max_llm_calls         # plan + judge batches + rank within budget
