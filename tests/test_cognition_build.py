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

    def _stmt(i, speaker, date, kind, about, role, facet, text, meta, pass_="t"):
        ext.execute("INSERT INTO statements(id, speaker, date, kind, about, role, facet, text, quote, epistemic, "
                    "condition, loc, meta, pass, item) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, speaker, date, kind, about, role, facet, text, "quote", "stated", "",
                     json.dumps({"unit_id": "u", "sent_id": f"s{i}"}), json.dumps(meta), pass_, speaker))

    _stmt(1, PB, "2020-06-01", "other", PA, "compares", "result", "B beats A on graphs",
          {"outcome": "citing_better", "category": "Graph Attention Models"})
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
    _stmt(8, PB, "2020-06-01", "self", PB, "proposes", "contribution", "proposes BiAttn",
          {"name": "BiAttn"})                                                   # makes biattn ambiguous
    _stmt(9, PD, "2022-02-01", "self", PD, "extends", "method", "extends BiAttn",
          {"mentions": [{"name": "BiAttn", "relation": "extends"}]})            # edge needs the adjudication
    # fact material: two independent papers state the same limitation, a third contradicts it
    _stmt(10, PA, "2021-01-01", "other", PB, "criticizes", "limitation",
          "GraphNet is slow on very large graphs", {})
    _stmt(11, PD, "2021-06-01", "other", PB, "criticizes", "limitation",
          "GraphNet is slow on large graphs", {})
    _stmt(12, PD, "2022-01-01", "other", PB, "background", "limitation",
          "GraphNet is not slow on large graphs at all", {})
    # COG-1010 repro material (10-10 state review): a same-speaker results-table version diff (13/14, the
    # FlowNet2 artifact — one paper's own cell in two text versions) and a same-speaker claim version
    # variance (15/16). Neither may produce a contested event: 13/14 never enter fact formation (results
    # pass, excluded by provenance), 15/16 are one voice (speaker-independence gate).
    _stmt(13, PB, "2020-06-01", "self", PB, "describes", "result", "GraphNet — Sintel: EPE: 0.78", {},
          pass_="results")
    _stmt(14, PB, "2021-01-15", "self", PB, "describes", "result", "GraphNet — Sintel: EPE: 3.41", {},
          pass_="results")
    _stmt(15, PD, "2021-02-01", "self", PD, "describes", "result",
          "GraphNet variants converge slowly on huge graphs", {})
    _stmt(16, PD, "2021-08-01", "self", PD, "describes", "result",
          "GraphNet variants converge quickly on huge graphs", {})
    ext.commit()
    ext.close()

    store.write_manifest("documents", {}, {})
    store.write_manifest("citations", {}, {})
    store.write_manifest("extract", {}, {})
    return store


def _chat(prompt, **kw):
    import json as _json
    if "OWNED" in prompt:                                    # method-identity adjudication
        assert "BiAttn" in prompt                            # the only ambiguous name in the fixture
        return _json.dumps({"paper": 1, "why": "proposal evidence"})   # most_common: PB(3) first
    if "canonical" in prompt:                                # category canonicalisation
        assert "Graph Attention Models" in prompt
        return _json.dumps({"canonical": "graph attention models", "umbrella": False})
    if "research FAMILY" in prompt:                          # family naming
        return _json.dumps({"name": "Graph Transformer Family", "keep": []})
    if "same underlying fact" in prompt:                     # fact relation
        if "0.78" in prompt or "3.41" in prompt:             # results-pass table version diff -> the LLM
            return _json.dumps({"relation": "opposite"})     # reads the two cells as contradicting
        if "converge" in prompt:                             # same-speaker version variance (15 vs 16)
            return _json.dumps({"relation": "opposite"})
        if "not slow" in prompt:
            return _json.dumps({"relation": "opposite"})
        if "slow on" in prompt:
            return _json.dumps({"relation": "same"})
        return _json.dumps({"relation": "unrelated"})
    raise AssertionError(f"unrouted prompt: {prompt[:80]}")


def test_cognition_build_materialises(env):
    from compilescholar.cognition import build as CB
    counts = CB.build(log=lambda *a: None, chat=_chat)
    assert counts["cocite"] == 1 and counts["reception"] == 3
    assert counts["author_link"] == 3 and counts["comparison_edge"] == 1
    assert counts["names"] == 3                              # fastgf, graphnet, biattn
    assert counts["names_filtered"] == {}                    # the fixture vocabulary is clean
    assert counts["mention_link"] == 4
    assert counts["mentions_ambiguous"] == 0 and counts["mentions_adjudicated"] == 1
    assert counts["identities"] == {"ok": 2, "adjudicated": 1}
    assert counts["lineage_edge"] == 3 and counts["lineage_hyper"] == 0
    assert counts["category_canon"] == 1 and counts["category_daily"] == 1
    # families: the lineage edges (valid_from 2022-01-01+) form one family {PB, PD}; the cocite pair
    # (PB, stub:xray) never does — stubs are boundary nodes, not family members; PA stays isolated
    assert counts["family_snapshots"] == 6 and counts["families"] == 6     # 2022-01 .. 2022-06, one family each
    assert counts["facts_contested"] >= 1
    con = env.connect("cognition", readonly=True)
    fams = con.execute("SELECT snapshot, members, name, named_by FROM family_snapshot ORDER BY snapshot").fetchall()
    assert min(s for s, *_ in fams) >= "2022-01-01"
    for snap, members, name, named_by in fams:
        assert sorted(json.loads(members)) == sorted([PB, PD])   # a 2-member family: no weak-link dropping
        assert name == "graph transformer family"
    assert counts["families_named"] == 6
    assert fams[-1][3] != "inherit" and all(f[3] == "inherit" for f in fams[:-1])  # only the latest is LLM-named
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
    assert ment[("biattn", PA)][5] == PB and ment[("biattn", PA)][6] == "adjudicated"
    assert json.loads(ment[("biattn", PA)][4]) == [[PA, 2], [PB, 3]]           # alias(2) vs proposal(3)
    assert not any(sid == f"{PA}@v1#s4" for (_, sid, *_ ) in
                   con.execute("SELECT name, sid FROM mention_link"))           # word-boundary discipline
    # lineage: the time-inconsistent edge is dropped, the valid ones carry §2.4 effective dates
    assert con.execute("SELECT canonical, umbrella FROM category_canon WHERE phrase='Graph Attention Models'"
                       ).fetchone() == ("graph attention models", 0)
    # facts: statements 10 and 11 form one fact (same), 12 contradicts it (opposite -> contested)
    f10 = con.execute("SELECT fact_id FROM fact_member WHERE statement_id=10").fetchone()[0]
    f11 = con.execute("SELECT fact_id FROM fact_member WHERE statement_id=11").fetchone()[0]
    f12 = con.execute("SELECT fact_id FROM fact_member WHERE statement_id=12").fetchone()[0]
    assert f10 == f11 and f12 != f10
    ev = con.execute("SELECT date, status FROM fact_status_event WHERE fact_id=? ORDER BY date, rowid",
                     (f10,)).fetchall()
    assert ("2021-01-01", "single-source") in ev                 # PA states it first
    assert ("2021-06-01", "established") in ev                   # PD (independent author) joins
    assert ("2022-01-01", "contested") in ev                     # the contradiction
    assert con.execute("SELECT count(*) FROM fact_member WHERE fact_id=? AND role='representative'",
                       (f10,)).fetchone()[0] == 1
    # COG-1010: results-pass table rows never enter fact formation (excluded by provenance), and a
    # same-speaker 'opposite' pair is not contested (one paper = one voice) — while the cross-author
    # contradiction above still is
    assert con.execute("SELECT count(*) FROM fact_member WHERE statement_id IN (13,14)").fetchone()[0] == 0
    f15 = con.execute("SELECT fact_id FROM fact_member WHERE statement_id=15").fetchone()[0]
    f16 = con.execute("SELECT fact_id FROM fact_member WHERE statement_id=16").fetchone()[0]
    assert f15 != f16                                        # 'opposite': the two versions never merge
    assert con.execute("SELECT count(*) FROM fact_status_event WHERE status='contested' AND fact_id IN (?,?)",
                       (f15, f16)).fetchone()[0] == 0
    assert con.execute("SELECT * FROM category_daily").fetchall() == [("graph attention models", "2020-06-01", 1)]
    edges = sorted(con.execute("SELECT child, parent, relation, kind, date, valid_from FROM lineage_edge"))
    assert edges == [(PD, PB, "extends", "self", "2022-01-01", "2022-01-01"),          # id6 self claim
                     (PD, PB, "extends", "self", "2022-02-01", "2022-02-01"),          # id9 via adjudicated identity
                     (PD, PB, "improves", "third", "2022-06-01", "2022-06-01")]        # id7 builds_on (child=about)
    con.close()


def test_cognition_incremental(env):
    from compilescholar.cognition import build as CB
    CB.build(log=lambda *a: None, chat=_chat)
    con = env.connect("cognition", readonly=True)
    w = {r[0]: r[1:] for r in con.execute("SELECT item, key, status FROM _work WHERE pass='cocite'")}
    con.close()
    CB.build(log=lambda *a: None, chat=_chat)            # nothing changed: no re-open
    con = env.connect("cognition", readonly=True)
    w2 = {r[0]: r[1:] for r in con.execute("SELECT item, key, status FROM _work WHERE pass='cocite'")}
    con.close()
    assert w == w2
    env.write_manifest("citations", {}, {"n": 1})        # citations data moved...
    env.write_manifest("extract", {}, {})                # ...extract re-registers against it (the real chain rebuilds)
    CB.build(log=lambda *a: None, chat=_chat)
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
    assert len(ss) == 7                                    # ids 1-5, 8, 13 are dated <= T (6, 7, 14-16 later)
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


# ---------------------------------------------------------------- shift screening (pure functions)
def test_fisher_and_bh():
    from compilescholar.cognition.build import bh_select, fisher_p
    assert fisher_p(4, 4, 4, 4) == 1.0
    assert fisher_p(0, 8, 8, 0) < 0.001                       # perfect flip
    assert fisher_p(8, 0, 0, 8) < 0.001
    assert fisher_p(3, 5, 4, 4) > 0.5                         # nothing there
    assert bh_select([0.001, 0.9, 0.02], q=0.05) == {0, 2}
    assert bh_select([0.9, 0.8]) == set()


def _vote(date, i, role="background", function="background", facet="contribution", category=None):
    return {"date": date, "role": role, "function": function, "facet": facet, "category": category,
            "speaker": f"s{i}", "quote": f"{role} vote {i} on {date}"}


def test_screen_became_component():
    from compilescholar.cognition.build import screen_shifts
    votes = ([_vote(f"2019-{m:02d}-01", m) for m in range(1, 9)]
             + [_vote(f"2021-{m:02d}-01", 100 + m, role="uses", function="tool") for m in range(1, 9)])
    cands = screen_shifts({"arxiv:x": votes}, {}, {})
    comp = [c for c in cands if c["facet"] == "became_component"]
    assert len(comp) == 1
    d = json.loads(comp[0]["direction"])
    assert d["early"] == 0.0 and d["late"] == 1.0 and d["p"] < 0.01
    ev = json.loads(comp[0]["evidence"])
    assert len(ev["late"]) == 3 and ev["early"] == []


def test_screen_recategorized_and_limitation_independence():
    from compilescholar.cognition.build import screen_shifts
    votes = ([_vote(f"2019-{m:02d}-01", m, category="graph models") for m in range(1, 5)]
             + [_vote(f"2021-{m:02d}-01", 100 + m, category="transformer methods") for m in range(1, 5)])
    cands = screen_shifts({"arxiv:x": votes}, {}, {})
    rc = [c for c in cands if c["facet"] == "recategorized"]
    assert len(rc) == 1 and json.loads(rc[0]["direction"])["from"] == "graph models"
    # limitation exposure needs two INDEPENDENT stating papers (no shared author key)
    lim = ([_vote(f"2019-{m:02d}-01", m) for m in range(1, 9)]
           + [_vote("2020-01-01", 50, facet="limitation"), _vote("2020-06-01", 51, facet="limitation")])
    same = {"s50": {"smith|j"}, "s51": {"smith|j"}}
    assert not [c for c in screen_shifts({"arxiv:y": lim}, same, {}) if c["facet"] == "limitation_exposed"]
    indep = {"s50": {"smith|j"}, "s51": {"wang|k"}}
    got = [c for c in screen_shifts({"arxiv:y": lim}, indep, {}) if c["facet"] == "limitation_exposed"]
    assert len(got) == 1 and got[0]["window_end"] == "2020-06-01"
    # too few votes: nothing screened
    assert screen_shifts({"arxiv:z": lim[:5]}, {}, {}) == []


# ---------------------------------------------------------------- identity vocab filter (COG-1010 DRIFT①)
def test_vocab_reject_rules():
    from compilescholar.cognition.build import vocab_reject
    # the audit's real contamination examples (audit_t2.json / report_t1t2.md §4) must be rejected
    for bad, rule in [("posterior sampling for reinforcement learning", "lowercase_phrase"),
                      ("privacy-preserving parameter tuning technique", "lowercase_phrase"),
                      ("a method that enables physically simulated characters to learn skills from videos",
                       "lowercase_phrase"),
                      ("Algorithm 2", "algorithm_n"), ("algorithm 12", "algorithm_n"),
                      ("UCF101", "dataset"), ("Scikit-learn", "dataset"), ("sklearn", "dataset"),
                      ("MovieBook Dataset", "dataset"),
                      ("Describable Textures Dataset (DTD)", "dataset"),
                      ("Stanford Question Answering Dataset (SQuAD)", "dataset"),
                      ("phrase localization benchmark", "dataset"),
                      ("Stanford Natural Language Inference corpus", "dataset")]:
        assert vocab_reject(bad) == rule, bad
    # real names must be kept: proper-name morphology (caps/digits/hyphens), short lowercase names, the
    # production names from the 2688fd0 family work (VQ-VAE's own ecosystem + the giant's name), and the
    # conventionally-lowercase 3-gram real methods (the narrowing rationale: 错杀比漏杀贵)
    for good in ["VQ-VAE", "GPT-4o", "Qwen3.8-27B", "k-means", "adaptive mcmc", "DAGGER", "PRM*",
                 "1-nearest sPRM", "CoveringBalls", "Optimal Probabilistic RoadMaps", "RRG algorithm",
                 "value iteration network", "sinusoidal representation networks", "model predictive control",
                 "Neural Discrete Representation Learning", "DVAE#", "GumBolt", "LPCNet",
                 "Objective perturbation", "Dataset Aggregation"]:   # capitalized surfaces: structure cannot
        assert vocab_reject(good) is None, good                     # judge borrowed-vs-own, so they stay
    assert vocab_reject("") is None and vocab_reject(None) is None


def test_name_vocab_filters_at_the_funnel():
    from collections import Counter
    from compilescholar.cognition.build import name_vocab
    rows_self = [(json.dumps({"name": "posterior sampling for reinforcement learning"}), "p1"),
                 (json.dumps({"name": "FastGF", "aliases": ["MovieBook Dataset", "fast gf"]}), "p1")]
    rows_other = [(json.dumps({"name": "Algorithm 3"}), "p2"),          # dies even earlier: norm_name's
                  (json.dumps({"name": "Visual Genome dataset"}), "p2"),  # genre-modifier strip leaves "3"
                  (json.dumps({"name": "GraphNet"}), "p2")]

    class _Ext:                                    # name_vocab's two queries are its whole interface
        def execute(self, q, *a):
            return iter(rows_self if "kind='self'" in q else rows_other)

    filt = {"counts": Counter(), "samples": []}
    v = name_vocab(_Ext(), filtered=filt)
    assert set(v) == {"fastgf", "fast gf", "graphnet"}          # proposals, aliases and third-party
    assert filt["counts"] == {"lowercase_phrase": 1, "dataset": 2}     # namings alike
    assert sorted(s for _, s in filt["samples"]) == sorted(
        ["posterior sampling for reinforcement learning", "MovieBook Dataset", "Visual Genome dataset"])


# ---------------------------------------------------------------- family partitioning (COG-1010)
def test_partition_families_giant_split():
    from compilescholar.cognition.build import partition_families
    # two 12-cliques joined by ONE weak bridge edge, plus a strong 2-node family; max_size 5 -> the cliques
    # are recursively re-clustered at doubled resolution on their INDUCED subgraphs (the bridge edge leaves
    # a community and must not reach the sub-run), the pair survives, the partition stays exact/deterministic
    def clique(p):
        return [(f"{p}{i}", f"{p}{j}") for i in range(12) for j in range(i + 1, 12)]
    edges = clique("c") + clique("d") + [("c0", "d0"), ("p1", "p2")]
    w = {e: (10 if e == ("p1", "p2") else 1) for e in edges}
    comms = partition_families(edges, w, resolution=1.0, max_size=5)
    assert sorted(x for c in comms for x in c) == sorted({x for e in edges for x in e})  # disjoint + complete
    assert all(len(c) <= 5 for c in comms)
    assert ["p1", "p2"] in comms                           # the small strong family is untouched
    assert comms == partition_families(edges, w, resolution=1.0, max_size=5)   # fixed seed: deterministic
    # no max_size pressure: the plain resolution-1 partition keeps the two cliques whole
    assert sorted(map(len, partition_families(edges, w, resolution=1.0, max_size=1000))) == [2, 12, 12]


def test_families_split_via_build(env, monkeypatch):
    from compilescholar.cognition import build as CB
    default_res, default_max = CB.FAMILY_RESOLUTION, CB.FAMILY_MAX
    monkeypatch.setattr(CB, "FAMILY_RESOLUTION", 1.5)
    monkeypatch.setattr(CB, "FAMILY_MAX", 1)               # the 2-member {PB,PD} family counts as "giant"
    counts = CB.build(log=lambda *a: None, chat=_chat)
    assert counts["families"] == 0                         # split to singletons -> below MIN_FAMILY
    # restore by setattr, never monkeypatch.undo(): the env fixture's CS_DATA patch would go with it and the
    # next build would gate against the REAL production manifests. The parameters are in the pass digest, so
    # the normal values re-open the pass and restore the family.
    monkeypatch.setattr(CB, "FAMILY_RESOLUTION", default_res)
    monkeypatch.setattr(CB, "FAMILY_MAX", default_max)
    counts = CB.build(log=lambda *a: None, chat=_chat)
    assert counts["families"] == 6
