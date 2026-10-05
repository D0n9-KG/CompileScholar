# -*- coding: utf-8 -*-
"""dfc.store, the stage contract (INTEGRATED-SYSTEM-1005 v2 §10.1). The probes of FITNESS-INFRA-COGNITION §1c are
regression tests here, each asserting the repaired behaviour:
  P1 code identity is the import closure (llm/client.py counts for extract)
  P2 a code change makes a rebuild redo the items, not just re-stamp the manifest
  P3 a params change re-opens the work instead of keeping stale rows
  P4 new upstream data (no config change) makes downstream `behind`, and only the new items are done
  P5 a failed item is never done; it is retried on the next build and gives up after MAX_ATTEMPTS
  P6 manifests are written atomically (a concurrent reader never sees half a file)
  P7 two builders of one stage: the second refuses (lock)
  P8 a stage whose block raises writes no manifest; consecutive failures trip the breaker"""
import importlib
import json
import threading

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path))
    from compilescholar.core import paths
    importlib.reload(paths)
    from compilescholar.dfc import store as S
    importlib.reload(S)
    return S


def _stand_up(store, *stages):
    for s in stages:
        store.write_manifest(s, {}, {})


def test_missing_upstream_refuses(store):
    with pytest.raises(RuntimeError):
        store.write_manifest("documents", {}, {})
    with pytest.raises(RuntimeError):
        with store.Run("documents", {}):
            pass


def test_upstream_reconfigured_makes_downstream_stale(store):
    store.write_manifest("papers", {"v": 1}, {"papers": 1})
    _stand_up(store, "documents", "citations")
    assert not store.status("citations")["stale"]
    store.write_manifest("papers", {"v": 2}, {"papers": 2})
    s = store.status("citations")
    assert s["stale"] and any("reconfigured" in w or "stale" in w for w in s["why"])
    with pytest.raises(RuntimeError):
        store.require_fresh("citations")


def test_transitive_staleness_from_code(store):
    _stand_up(store, "papers", "documents", "citations", "extract")
    m = store.read_manifest("documents")
    m["code"] = {"documents/units.py": "old"}
    json.dump(m, open(store.root() / "manifests" / "documents.json", "w"))
    assert store.status("documents")["stale"]
    assert store.status("citations")["stale"] and store.status("extract")["stale"]


def test_p1_code_identity_is_the_import_closure(store):
    code = store.code_hashes(store.ENTRY["extract"])
    for f in ("llm/client.py", "llm/jsonparse.py", "compile/skeleton/proposes.py", "extract/self_pass.py",
              "documents/units.py"):
        assert f in code, f
    assert "llm/embedding.py" in store.code_hashes(store.ENTRY["index"])
    assert "compile/skeleton/bib.py" in store.code_hashes(store.ENTRY["documents"])


def _items_build(store, stage, params, items, fn, rebuild=False):
    """A minimal stage: one pass whose items are (id, fingerprint); fn(id) -> True ok / False fail."""
    done = []
    with store.Run(stage, params, rebuild=rebuild) as run:
        w = run.work("main")
        for i in w.todo(items):
            if fn(i):
                done.append(i)
                w.ok(i)
            else:
                w.fail(i, "boom")
        w.sweep([x[0] for x in items], lambda i: None)
        run.finish({"n": len(items)})
    return done


def test_p2_p3_config_change_redoes_items(store):
    items = [("a", "1"), ("b", "1")]
    assert _items_build(store, "papers", {"v": 1}, items, lambda i: True) == ["a", "b"]
    assert _items_build(store, "papers", {"v": 1}, items, lambda i: True) == []          # nothing changed
    assert sorted(_items_build(store, "papers", {"v": 2}, items, lambda i: True)) == ["a", "b"]  # params changed
    # code change: simulate by poisoning the recorded keys' digest source (the digest includes the code closure)
    orig = store.code_hashes
    store.code_hashes = lambda entries: {**orig(entries), "corpus/papers.py": "changed"}
    try:
        assert sorted(_items_build(store, "papers", {"v": 2}, items, lambda i: True)) == ["a", "b"]
    finally:
        store.code_hashes = orig


def test_p4_new_upstream_data_is_behind_and_incremental(store):
    _items_build(store, "papers", {}, [("a", "1")], lambda i: True)
    _items_build(store, "documents", {}, [("a", "1")], lambda i: True)
    assert not store.status("documents")["behind"]
    _items_build(store, "papers", {}, [("a", "1"), ("b", "1")], lambda i: True)   # data grew, config same
    st = store.status("documents")
    assert st["behind"] and not st["stale"]
    with pytest.raises(RuntimeError):
        store.require_fresh("documents")
    # downstream recomputes only the new item (its fingerprint names the upstream item it depends on)
    assert _items_build(store, "documents", {}, [("a", "1"), ("b", "1")], lambda i: True) == ["b"]
    assert not store.status("documents")["behind"]


def test_p5_failures_are_retried_then_given_up(store):
    calls = []

    def flaky(i):
        calls.append(i)
        return False

    for _ in range(store.MAX_ATTEMPTS):
        _items_build(store, "papers", {}, [("a", "1")], flaky)
        m = store.read_manifest("papers")
        assert not m["complete"]                     # a failed item is never "done"
    assert calls == ["a"] * store.MAX_ATTEMPTS
    _items_build(store, "papers", {}, [("a", "1")], flaky)
    assert calls == ["a"] * store.MAX_ATTEMPTS        # gave up: not retried under the same key
    assert store.read_manifest("papers")["work"].get("gave_up") == 1
    _items_build(store, "papers", {}, [("a", "2")], lambda i: True)   # input changed -> tried again
    assert store.read_manifest("papers")["complete"]


def test_p5_sweep_removes_items_that_left(store):
    removed = []
    with store.Run("papers", {}) as run:
        w = run.work("main")
        for i in w.todo([("a", "1"), ("b", "1")]):
            w.ok(i)
        run.finish({})
    with store.Run("papers", {}) as run:
        w = run.work("main")
        w.todo([("a", "1")])
        assert w.sweep(["a"], removed.append) == 1
        run.finish({})
    assert removed == ["b"]


def test_p6_manifest_is_atomic(store):
    _stand_up(store, "papers")
    bad = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                store.read_manifest("papers")
            except Exception as e:      # noqa: BLE001 — any read failure is the bug
                bad.append(e)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(150):
        store.write_manifest("papers", {"i": i}, {"n": i})
    stop.set()
    t.join()
    assert not bad


def test_p7_second_builder_refuses(store):
    with store.Run("papers", {}):
        with pytest.raises(store.StageLocked):
            with store.Run("papers", {}):
                pass


def test_p8_failed_block_writes_no_manifest_and_breaker_trips(store):
    with pytest.raises(ValueError):
        with store.Run("papers", {}) as run:
            run.work("main").todo(["a"])
            raise ValueError("crash")
    assert store.read_manifest("papers") is None
    with pytest.raises(store.StageAborted):
        with store.Run("papers", {}) as run:
            w = run.work("main", breaker=3)
            for i in w.todo([str(k) for k in range(10)]):
                w.fail(i, "server down")
    assert store.read_manifest("papers") is None


def test_rebuild_swaps_in_a_clean_file(store):
    with store.Run("papers", {}) as run:
        run.con.execute("CREATE TABLE t(x)")
        run.con.execute("INSERT INTO t VALUES (1)")
        run.finish({})
    with store.Run("papers", {}, rebuild=True) as run:
        assert not run.con.execute("SELECT name FROM sqlite_master WHERE name='t'").fetchall()
        run.con.execute("CREATE TABLE t(x)")
        run.con.execute("INSERT INTO t VALUES (2)")
        run.finish({})
    con = store.connect("papers", readonly=True)
    assert con.execute("SELECT x FROM t").fetchall() == [(2,)]


def test_old_store_without_bookkeeping_needs_rebuild(store):
    con = store.connect("papers")
    con.execute("CREATE TABLE legacy(x)")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="rebuild"):
        with store.Run("papers", {}):
            pass
    with store.Run("papers", {}, rebuild=True) as run:
        run.finish({})
