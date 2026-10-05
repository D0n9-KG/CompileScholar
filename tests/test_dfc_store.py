# -*- coding: utf-8 -*-
"""dfc.store: manifests chain stages; rebuilding upstream or changing a stage's code makes downstream stale."""
import importlib

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path))
    from compilescholar.dfc import store as S
    importlib.reload(S)
    return S


def test_missing_upstream_refuses(store):
    with pytest.raises(RuntimeError):
        store.write_manifest("citations", {}, {})


def test_fresh_then_stale_after_upstream_rebuild(store):
    store.write_manifest("papers", {"v": 1}, {"papers": 1})
    store.write_manifest("citations", {}, {"docs": 1})
    assert not store.status("citations")["stale"]
    store.write_manifest("papers", {"v": 2}, {"papers": 2})
    s = store.status("citations")
    assert s["stale"] and any("papers rebuilt" in w for w in s["why"])
    with pytest.raises(RuntimeError):
        store.require_fresh("citations")


def test_transitive_staleness(store):
    store.write_manifest("papers", {"v": 1}, {})
    store.write_manifest("citations", {}, {})
    store.write_manifest("extract", {}, {})
    m = store.read_manifest("citations")
    m["code"] = {"citations/markdown.py": "old"}
    import json
    json.dump(m, open(store.root() / "manifests" / "citations.json", "w"))
    assert store.status("citations")["stale"]
    assert store.status("extract")["stale"]
