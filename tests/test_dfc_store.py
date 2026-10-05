# -*- coding: utf-8 -*-
"""dfc.store: manifests chain stages; rebuilding upstream or changing a stage's code makes downstream stale."""
import importlib
import json

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path))
    from compilescholar.dfc import store as S
    importlib.reload(S)
    return S


def test_missing_upstream_refuses(store):
    with pytest.raises(RuntimeError):
        store.write_manifest("documents", {}, {})


def test_fresh_then_stale_after_upstream_rebuild(store):
    store.write_manifest("papers", {"v": 1}, {"papers": 1})
    store.write_manifest("documents", {}, {"docs": 1})
    store.write_manifest("citations", {}, {"docs": 1})
    assert not store.status("citations")["stale"]
    store.write_manifest("papers", {"v": 2}, {"papers": 2})
    s = store.status("citations")
    assert s["stale"] and any("papers rebuilt" in w for w in s["why"])
    with pytest.raises(RuntimeError):
        store.require_fresh("citations")


def test_transitive_staleness(store):
    store.write_manifest("papers", {"v": 1}, {})
    store.write_manifest("documents", {}, {})
    store.write_manifest("citations", {}, {})
    store.write_manifest("extract", {}, {})
    m = store.read_manifest("documents")
    m["code"] = {"documents/units.py": "old"}
    json.dump(m, open(store.root() / "manifests" / "documents.json", "w"))
    assert store.status("documents")["stale"]
    assert store.status("citations")["stale"] and store.status("extract")["stale"]
