# -*- coding: utf-8 -*-
"""grow.plan: typed gap diagnosis from corpus statistics only."""
import importlib
import json

import pytest

pytestmark = pytest.mark.skip(
    reason="grow.plan reads the pre-C arXiv-keyed derived stores; it is re-wired onto paper_id keys in phase D "
           "(INTEGRATED-SYSTEM-1005 §12), and this test returns with it")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path))
    from compilescholar.dfc import store
    importlib.reload(store)
    for s in ("papers", "documents", "citations", "extract"):
        store.connect(s).close()
    store.write_manifest("papers", {}, {})
    d = store.connect("documents")
    d.execute("CREATE TABLE docs(arxiv_id TEXT PRIMARY KEY, source TEXT, v1_date TEXT, n_chars INT, n_units INT, "
              "n_tables INT, raw_z BLOB, units_z BLOB)")
    for a in ("S1", "S2"):
        d.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)", (a, "md", "2021-01-01", 1, 1, 0, b"", b""))
    d.commit()
    store.write_manifest("documents", {}, {})
    c = store.connect("citations")
    c.execute("CREATE TABLE cites(sentence_id INT, citing TEXT, date TEXT, key TEXT, cited TEXT, n_group INT)")
    rows = [(i, f"C{i}", "2021-03-01", "k", "paper:U", 1) for i in range(4)]          # U cited by 4, not held
    rows += [(10 + i, f"C{i}", "2021-03-01", "k", "paper:S1", 1) for i in range(6)]   # S1 cited by 6, only T1
    rows += [(20 + i, f"C{i}", "2021-03-01", "k", "stub:some old book", 1) for i in range(3)]
    c.executemany("INSERT INTO cites VALUES (?,?,?,?,?,?)", rows)
    c.commit()
    store.write_manifest("citations", {}, {})
    e = store.connect("extract")
    e.executescript("CREATE TABLE scope(arxiv_id TEXT PRIMARY KEY, deep INT);"
                    "CREATE TABLE tiers(arxiv_id TEXT PRIMARY KEY, tier TEXT, stats TEXT);")
    e.executemany("INSERT INTO scope VALUES (?,?)", [("S1", 0), ("S2", 1)])
    e.executemany("INSERT INTO tiers VALUES (?,?,?)", [("S1", "T1", "{}"), ("S2", "T2", "{}")])
    e.commit()
    store.write_manifest("extract", {}, {})
    from compilescholar.grow import plan
    importlib.reload(plan)
    return plan


def test_diagnose_types(env):
    p = env.diagnose("2022-06-01")
    assert [x["arxiv_id"] for x in p["unread_member"]] == ["U"]
    assert [x["arxiv_id"] for x in p["shallow"]] == ["S1"]
    assert p["unresolved"][0]["title"] == "some old book" and p["unresolved"][0]["action"] == "record_only"
    assert p["frontier"]["n_scope_papers_stale"] == 1          # S1's newest citer is 2021-03, > 6 months before now
