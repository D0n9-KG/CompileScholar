# -*- coding: utf-8 -*-
"""cognition: identity (method -> proposing paper), lineage (self vs third-party, time consistency, n-ary combines),
facts (support set, independence by author overlap, status), shift events, qualitative comparisons — on a store built
directly from statements (no LLM)."""
import importlib
import json

import pytest

PAPERS = {  # id: (date, authors)
    "A": ("2019-01-01", ["Smith", "Lee"]),
    "B": ("2020-01-01", ["Wang"]),
    "C": ("2020-06-01", ["Smith", "Lee", "Kim"]),     # same group as A
    "D": ("2021-01-01", ["Zhao"]),
    "E": ("2021-06-01", ["Garcia"]),
    "F": ("2018-01-01", ["Old"]),
}


def st(speaker, about, kind, role, facet, text, quote=None, function=None, meta=None, group=(), date=None):
    return dict(speaker=speaker, date=date or PAPERS[speaker][0], kind=kind, about=f"paper:{about}", role=role,
                facet=facet, text=text, quote=quote or text, target=None, grp=json.dumps([f"paper:{g}" for g in group]),
                function=function, meta=json.dumps(meta or {}), model="t", prompt_sha="t")


ROWS = [
    st("A", "A", "self", "proposes", "contribution", "proposes GraphNet",
       "We propose GraphNet, a new model for graphs.", meta={"name": "GraphNet", "aliases": [], "generic": False}),
    st("B", "A", "other", "improves", "method", "GraphNet uses attention", function="basis",
       meta={"sentence_id": 1, "name": "GraphNet", "category": "graph attention models"}),
    st("C", "A", "other", "background", "limitation", "GraphNet is slow on large graphs", function="background",
       meta={"sentence_id": 2, "category": "graph attention models"}),
    st("D", "A", "other", "uses", "limitation", "GraphNet is slow on large graphs", function="tool",
       meta={"sentence_id": 3, "category": "graph attention models"}),
    st("E", "A", "other", "uses", "limitation", "GraphNet slow on very large graphs", function="tool",
       meta={"sentence_id": 4, "category": "graph attention models"}),
    st("D", "B", "other", "compares", "method", "FastNet improves GraphNet", function="baseline",
       meta={"sentence_id": 5, "builds_on": "GraphNet", "builds_on_relation": "improves", "outcome": "citing_better"}),
    # time-inconsistent claim: F (2018) "extends" A (2019) -> dropped
    st("F", "A", "other", "extends", "method", "extends GraphNet", function="basis", meta={"sentence_id": 6}),
    # n-ary combines: E combines A and B in one sentence
    st("E", "B", "other", "combines", "method", "combines FastNet", function="basis", meta={"sentence_id": 7}),
    st("E", "A", "other", "combines", "method", "combines GraphNet", function="basis", meta={"sentence_id": 7}),
]


@pytest.fixture()
def view(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path))
    from compilescholar.dfc import store
    importlib.reload(store)
    from compilescholar.corpus import papers as PP
    importlib.reload(PP)
    # the derived "papers" stage is retired (phase C-1: the registry replaces it); this legacy fixture keeps its
    # arXiv-keyed fake store at an explicit path and injects it into AsOf until cognition is re-wired in phase D
    con = PP.connect(tmp_path / "papers.sqlite")
    for pid, (d, au) in PAPERS.items():
        con.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, d, f"Paper {pid}", f"paper {pid}", "abs", json.dumps(au), au[0].lower(), "cs.LG", "cs.LG",
                     None, "t"))
    con.commit()
    con.close()
    from compilescholar.extract.schema import DDL
    ext = store.connect("extract")
    ext.executescript(DDL)
    for r in ROWS:
        ext.execute("INSERT INTO statements(speaker,date,kind,about,role,facet,text,quote,target,grp,function,meta,"
                    "model,prompt_sha) VALUES (:speaker,:date,:kind,:about,:role,:facet,:text,:quote,:target,:grp,"
                    ":function,:meta,:model,:prompt_sha)", r)
    ext.commit()
    cit = store.connect("citations")
    cit.executescript("CREATE TABLE cites(sentence_id INT, citing TEXT, date TEXT, key TEXT, cited TEXT, n_group INT);"
                      "CREATE TABLE entries(citing TEXT, key TEXT, raw TEXT, cited TEXT, method TEXT, title TEXT, year INT);")
    cit.commit()
    from compilescholar.cognition.asof import AsOf
    papers_ro = store.read_only(tmp_path / "papers.sqlite")
    return lambda T: AsOf(T, papers=papers_ro)


def test_identity_anchors_method_to_proposer(view):
    from compilescholar.cognition.identity import Identity
    ident = Identity(view("2022-01-01"))
    assert ident.resolve("GraphNet") == "paper:A" and ident.resolve("graphnet") == "paper:A"
    assert ident.resolve("Transformer") is None


def test_lineage_self_third_time_and_combines(view):
    from compilescholar.cognition import lineage as L
    E = L.edges(view("2022-01-01"))
    by = {(e["child"], e["parent"], e["relation"]): e for e in E["edges"]}
    assert ("paper:B", "paper:A", "improves") in by
    assert len(by[("paper:B", "paper:A", "improves")]["self"]) == 1      # B says so about itself
    assert len(by[("paper:B", "paper:A", "improves")]["third"]) == 1     # D says FastNet(B) improves GraphNet(A)
    assert E["dropped_time_inconsistent"] == 1                            # F (2018) cannot extend A (2019)
    assert E["combines"] == [{"child": "paper:E", "parents": ["paper:A", "paper:B"], "sentence_id": 7}]
    assert L.edges(view("2020-12-31"))["combines"] == []                  # E not visible yet


def test_facts_independence_and_status(view):
    from compilescholar.cognition import facts as FA
    fx = FA.facts(view("2022-01-01"), ["paper:A"], facets=("limitation",))
    slow = [f for f in fx if "slow" in f["text"]][0]
    assert slow["n_papers"] == 3                     # C, D, E
    assert slow["n_independent"] == 3                # C shares authors with A, not with D/E
    assert slow["status"] == "established"           # one member only -> not consensus
    early = FA.facts(view("2020-12-31"), ["paper:A"], facets=("limitation",))
    assert early and early[0]["status"] == "single-source" and early[0]["first_seen"] == "2020-06-01"


def test_independence_merges_same_group(view):
    from compilescholar.cognition.facts import independent
    assert independent(view("2022-01-01"), {"A", "C"}) == 1
    assert independent(view("2022-01-01"), {"A", "D"}) == 2


def test_shift_events_and_comparisons(view):
    from compilescholar.cognition import comparisons as C
    from compilescholar.cognition import shifts as SH
    ev = SH.events(view("2022-01-01"), "paper:A")
    assert any(e["type"] == "limitation_exposed" and e["date"] == "2021-01-01" for e in ev)
    assert not any(e["type"] == "limitation_exposed" for e in SH.events(view("2020-12-31"), "paper:A"))
    g = C.compared_with(view("2022-01-01"), "paper:B")
    assert g["n_compared_by"] == 1 and g["outcomes"] == {"citing_better": 1}
