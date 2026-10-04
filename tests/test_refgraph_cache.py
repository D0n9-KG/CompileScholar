# -*- coding: utf-8 -*-
"""W1-9: refgraph never caches an empty route after a transient failure; a definitive empty is cached; a non-empty
result is always cached. No network: the per-source functions are replaced."""
from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture()
def rg(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_REFGRAPH_CACHE", str(tmp_path))
    import compilescholar.sources.refgraph as R
    R = importlib.reload(R)
    monkeypatch.setattr(R, "_s2_cooling", lambda: False)
    monkeypatch.setattr(R, "crossref_references", lambda t: [])
    monkeypatch.setattr(R, "s2_references", lambda t: [])
    yield R
    importlib.reload(R)


def _routes(tmp_path):
    return [p for p in os.listdir(tmp_path) if p.startswith("route_")]


def test_transient_failure_not_cached(rg, tmp_path, monkeypatch):
    def failing(title, arxiv_id=None, allow_search=False):
        rg._mark_transient()
        return []
    monkeypatch.setattr(rg, "openalex_references", failing)
    refs, src = rg.references("Some paper title")
    assert refs == [] and src == "none"
    assert _routes(tmp_path) == []


def test_definitive_empty_cached(rg, tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "openalex_references", lambda title, arxiv_id=None, allow_search=False: [])
    rg.references("Another paper")
    assert len(_routes(tmp_path)) == 1


def test_deadline_cut_not_cached(rg, tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "openalex_references", lambda title, arxiv_id=None, allow_search=False: [])
    rg.references("Late paper", deadline=0.0)
    assert _routes(tmp_path) == []


def test_results_cached_even_after_transient(rg, tmp_path, monkeypatch):
    def partial(title, arxiv_id=None, allow_search=False):
        rg._mark_transient()
        return [{"title": f"Ref {i}", "year": 2020, "date": None, "abstract": "", "ids": {}} for i in range(6)]
    monkeypatch.setattr(rg, "openalex_references", partial)
    refs, src = rg.references("Has refs")
    assert len(refs) == 6 and len(_routes(tmp_path)) == 1
