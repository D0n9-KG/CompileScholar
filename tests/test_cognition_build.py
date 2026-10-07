# -*- coding: utf-8 -*-
"""Phase D①: the cognition stage (deterministic materialisations: cocite / reception_daily / author_link /
comparison_edge) and the rewired AsOf (registry visibility by first_hi, alias canonicalisation, schema v2
statements, version-dated citations)."""
from __future__ import annotations

import importlib
import json
import sqlite3

import pytest

PA = "arxiv:1901.00001"      # visible since 2019-01-01
PB = "doi:10.1/b"            # visible since 2020-01-15
PC = "arxiv:1901.00002"      # merged into PA
PD = "title:some year paper|2021"   # year precision: visible only at its hi (2021-12-31) — leak-safe


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path / "data"))
    from compilescholar.core import paths
    importlib.reload(paths)
    from compilescholar.dfc import store
    importlib.reload(store)
    from compilescholar.library import store as LS
    importlib.reload(LS)
    from compilescholar.citations.build import DDL as CIT_DDL
    from compilescholar.extract.schema import DDL as EXT_DDL

    reg = LS.connect()
    now = "2026-10-07T00:00:00"
    for pid, st, hi, pr in [(PA, "active", "2019-01-01", "day"), (PB, "active", "2020-01-15", "day"),
                            (PC, "merged", "2019-02-01", "day"), (PD, "active", "2021-12-31", "year")]:
        reg.execute("INSERT INTO papers(paper_id, status, title, first_hi, first_precision, first_source, "
                    "first_kind, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (pid, st, f"Paper {pid}", hi, pr, "test", "arxiv_v", now))
    reg.execute("INSERT INTO aliases VALUES (?,?,?,?)", (PC, PA, "test", now))
    reg.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                (PA, "crossref", None, None, "Journal title of A", "journaltitleofa", "journal abstract", "", "J"))
    reg.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                (PA, "arxiv", 1, "2019-01-01", "ArXiv title of A", "arxivtitleofa", "arxiv abstract", "cs.LG", ""))
    reg.execute("INSERT INTO authors VALUES (?,?,?)", (PA, "arxiv", json.dumps([{"name": "J. Smith", "surname": "Smith"}])))
    reg.execute("INSERT INTO authors VALUES (?,?,?)", (PB, "arxiv", json.dumps([{"name": "J. Smith", "surname": "Smith"}])))
    reg.execute("INSERT INTO authors VALUES (?,?,?)", (PD, "arxiv", json.dumps([{"name": "Ann Other", "surname": "Other"}])))
    reg.commit()
    reg.close()

    cit = store.connect("citations")
    cit.executescript(CIT_DDL)
    # sentence 1 (PA, 2019-06-01) cites PB and a stub together; sentence 2 (PB, 2020-06-01) cites PA
    cites = [(1, PA, "2019-06-01", 1, "b0", PB, 2, 0), (1, PA, "2019-06-01", 1, "b1", "stub:xray", 2, 0),
             (2, PB, "2020-06-01", 1, "b0", PA, 1, 1)]
    cit.executemany("INSERT INTO cites VALUES (?,?,?,?,?,?,?,?)", cites)
    cit.executemany("INSERT INTO entries(citing, version, key, raw, title, year, doi, arxiv, cited, method) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(PA, 1, "b0", "raw b0", "Paper B", 2020, None, None, PB, "entry_doi"),
                     (PA, 1, "b1", "raw b1", "X-ray", 2019, None, None, "stub:xray", "stub"),
                     (PA, 1, "b2", "raw b2", "Year paper", 2021, None, None, PD, "registry_title"),
                     (PB, 1, "b0", "raw", "Paper A", 2019, None, None, PA, "registry_title")])
    cit.commit()
    cit.close()

    import zlib
    from compilescholar.documents.build import DDL as DOC_DDL
    docs = store.connect("documents")
    docs.executescript(DOC_DDL)

    def _fastz(sents):
        d = {"title": "t", "abstract": "",
             "units": [{"uid": "u1", "kind": "para", "section": "", "text": " ".join(s[1] for s in sents)}],
             "sentences": [{"sid": s[0], "unit": "u1", "text": s[1]} for s in sents], "entries": {}, "cites": []}
        return zlib.compress(json.dumps(d).encode())

    def _doc(key, pid, ver, date, sents):
        docs.execute("INSERT INTO docs(key,paper_id,version,sha256,source,text_date,date_precision,n_units,"
                     "n_sentences,n_entries,n_cites,n_tables,fast_z,careful_z) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (key, pid, ver, "sha-" + key, "arxiv_nas", date, "day", 1, len(sents), 0, 0, 0,
                      _fastz(sents), None))

    _doc(f"{PA}@v1", PA, 1, "2019-06-01",
         [(f"{PA}@v1#s1", "We propose FastGF, an efficient graph transformer."),
          (f"{PA}@v1#s2", "GraphNet improves the state of the art."),
          (f"{PA}@v1#s3", "BiAttn is compared as a baseline."),
          (f"{PA}@v1#s4", "The FastGFnet variant is also tested.")])
    _doc(f"{PB}@v1", PB, 1, "2020-06-01", [(f"{PB}@v1#s1", "GraphNet is our proposed method.")])
    docs.commit()
    docs.close()

    ext = store.connect("extract")
    ext.executescript(EXT_DDL)

    def _stmt(i, speaker, date, kind, about, role, facet, text, meta):
        ext.execute("INSERT INTO statements(id, speaker, date, kind, about, role, facet, text, quote, epistemic, "
                    "condition, loc, meta, pass, item) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, speaker, date, kind, about, role, facet, text, "quote", "stated", "",
                     json.dumps({"unit_id": "u", "sent_id": f"s{i}"}), json.dumps(meta), "t", speaker))

    _stmt(1, PB, "2020-06-01", "other", PA, "compares", "result", "B beats A on graphs",
          {"outcome": "citing_better"})
    _stmt(2, PA, "2019-06-01", "self", PA, "proposes", "contribution", "proposes M", {})
    _stmt(3, PA, "2019-06-01", "self", PA, "proposes", "contribution", "proposes FastGF",
          {"name": "FastGF", "aliases": ["BiAttn"]})
    _stmt(4, PB, "2020-06-01", "self", PB, "proposes", "contribution", "proposes GraphNet", {"name": "GraphNet"})
    _stmt(5, PA, "2019-06-01", "self", PA, "extends", "method", "extends GraphNet",
          {"mentions": [{"name": "GraphNet", "relation": "extends"}]})          # time-inconsistent (PA<PB)
    _stmt(6, PD, "2022-01-01", "self", PD, "extends", "method", "extends GraphNet",
          {"mentions": [{"name": "GraphNet", "relation": "extends"}]})          # valid self edge
    _stmt(7, PB, "2022-06-01", "other", PD, "uses", "method", "PD improves GraphNet",
          {"builds_on": "GraphNet", "builds_on_relation": "improves"})           # third-party edge
    _stmt(8, PB, "2020-06-01", "self", PB, "proposes", "contribution", "proposes AttnNet",
          {"name": "BiAttn"})                                                   # makes attnet ambiguous
    ext.commit()
    ext.close()

    store.write_manifest("documents", {}, {})
    store.write_manifest("citations", {}, {})
    store.write_manifest("extract", {}, {})
    return store


def test_cognition_build_materialises(env):
    from compilescholar.cognition import build as CB
    counts = CB.build(log=lambda *a: None)
    assert counts["cocite"] == 1 and counts["reception"] == 3
    assert counts["author_link"] == 3 and counts["comparison_edge"] == 1
    assert counts["names"] == 3                              # fastgf, graphnet, biattn
    assert counts["mention_link"] == 4 and counts["mentions_ambiguous"] == 1
    assert counts["lineage_edge"] == 2 and counts["lineage_hyper"] == 0
    con = env.connect("cognition", readonly=True)
    a, b, day, n = con.execute("SELECT * FROM cocite").fetchone()
    assert {a, b} == {PB, "stub:xray"} and day == "2019-06-01" and n == 1     # ordered pair, one co-citation
    rec19 = dict(con.execute("SELECT cited, n FROM reception_daily WHERE day='2019-06-01'"))
    assert rec19 == {PB: 1, "stub:xray": 1}                                    # what sentence 1 cited
    assert con.execute("SELECT count(*) FROM reception_daily WHERE cited=? AND day='2020-06-01'", (PA,)).fetchone()[0] == 1
    keys = {k for (k,) in con.execute("SELECT DISTINCT author_key FROM author_link")}
    assert "smith|j" in keys and "other|a" in keys
    assert con.execute("SELECT count(*) FROM author_link WHERE author_key='smith|j'").fetchone()[0] == 2  # A and B
    row = con.execute("SELECT a, b, outcome, date FROM comparison_edge").fetchone()
    assert row == (PB, PA, "citing_better", "2020-06-01")
    # mentions: own/trivial uses resolved, ambiguous kept with candidates, substring "FastGFnet" not a mention
    ment = {(r[0], r[2]): r for r in con.execute(
        "SELECT name, sid, citing, date, candidates, paper_id, status FROM mention_link")}
    assert ("fastgf", PA) in ment and ment[("fastgf", PA)][5] == PA and ment[("fastgf", PA)][6] == "ok"
    assert ("graphnet", PA) in ment and ment[("graphnet", PA)][5] == PB       # mention in PA cites PB's method
    assert ("graphnet", PB) in ment                                            # self-mention kept
    assert ment[("biattn", PA)][5] is None and ment[("biattn", PA)][6] == "ambiguous"
    assert json.loads(ment[("biattn", PA)][4]) == [[PA, 2], [PB, 3]]           # alias(2) vs proposal(3)
    assert not any(sid == f"{PA}@v1#s4" for (_, sid, *_ ) in
                   con.execute("SELECT name, sid FROM mention_link"))           # word-boundary discipline
    # lineage: the time-inconsistent edge is dropped, the valid ones carry §2.4 effective dates
    edges = sorted(con.execute("SELECT child, parent, relation, kind, date, valid_from FROM lineage_edge"))
    assert edges == [(PD, PB, "extends", "self", "2022-01-01", "2022-01-01"),          # id6 self claim
                     (PD, PB, "improves", "third", "2022-06-01", "2022-06-01")]        # id7 builds_on (child=about)
    con.close()


def test_cognition_incremental(env):
    from compilescholar.cognition import build as CB
    CB.build(log=lambda *a: None)
    con = env.connect("cognition", readonly=True)
    w = {r[0]: r[1:] for r in con.execute("SELECT item, key, status FROM _work WHERE pass='cocite'")}
    con.close()
    CB.build(log=lambda *a: None)                        # nothing changed: no re-open
    con = env.connect("cognition", readonly=True)
    w2 = {r[0]: r[1:] for r in con.execute("SELECT item, key, status FROM _work WHERE pass='cocite'")}
    con.close()
    assert w == w2
    env.write_manifest("citations", {}, {"n": 1})        # citations data moved...
    env.write_manifest("extract", {}, {})                # ...extract re-registers against it (the real chain rebuilds)
    CB.build(log=lambda *a: None)
    con = env.connect("cognition", readonly=True)
    w3 = con.execute("SELECT key FROM _work WHERE pass='cocite' AND item='all'").fetchone()[0]
    wa = con.execute("SELECT key FROM _work WHERE pass='authors' AND item='all'").fetchone()[0]
    con.close()
    assert w3 != w["all"][0]


def test_asof_registry_visibility(env):
    from compilescholar.cognition.asof import AsOf
    v = AsOf("2021-06-01")
    p = v.paper(PA)
    assert p and p["title"] == "ArXiv title of A" and p["date"] == "2019-01-01"   # arxiv record wins
    assert p["authors"] == [{"name": "J. Smith", "surname": "Smith"}] and p["ids"] == {}
    assert v.paper(PB) is not None
    assert v.paper(PD) is None                            # year precision: hi 2021-12-31 > T -> not yet visible
    assert AsOf("2021-12-31").paper(PD) is not None       # ...visible exactly at its hi (leak-safe)
    assert v.canonical(PC) == PA and v.paper(PC) == p     # a merged id follows its alias
    assert v.visible("stub:xray") is True                 # stubs are boundary nodes, always visible
    assert v.visible(PA) and not v.visible("arxiv:9999.99999")


def test_asof_statements_and_citations(env):
    from compilescholar.cognition.asof import AsOf
    v = AsOf("2020-12-31")
    ss = v.statements()
    assert len(ss) == 6                                    # ids 1-5 and 8 are dated <= T (6 and 7 are 2022)
    assert all(isinstance(s["loc"], dict) and isinstance(s["meta"], dict) for s in ss)
    assert v.statements(about=PA, kind="other")[0]["text"] == "B beats A on graphs"
    v19 = AsOf("2019-12-31")
    assert len(v19.statements()) == 3                      # the 2019 statements only
    assert v.cited_by(PA) == [(PB, "2020-06-01", 2)]       # sentence 2 only
    assert v19.cited_by(PB) == [(PA, "2019-06-01", 1)]
    refs = v.references(PA)
    assert PB in refs and "stub:xray" in refs and PD not in refs   # PD invisible at T -> filtered
    assert v.references("arxiv:9999.99999") == []          # an invisible paper has no references
    assert set(v.co_cited(1)) == {PB, "stub:xray"}
