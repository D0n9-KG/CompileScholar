# -*- coding: utf-8 -*-
"""End-to-end invariants of the DFC data flow on a synthetic corpus (DESIGN §12a), no network, LLM stubbed:
papers -> citations -> extract -> cognition -> tools.

  1. every statement's quote is a substring of its source text (citation sentence or the speaker's own text);
  2. as_of monotone: nothing dated > T is visible at T; the visible set only grows with T;
  3. every id a tool returns exists in papers at as_of, or is a stub;
  4. a downstream stage refuses to run on a stale upstream."""
import importlib
import json

import pytest

TAIL = (" Experiments on several standard benchmarks show consistent gains over strong baselines in both accuracy and "
        "training cost.")
ABS = {
    "2001.00001": ("Attention is all you need for graphs",
                   "We propose GraphFormer, a new attention model for graph data. It outperforms message passing." + TAIL),
    "2101.00002": ("Faster graph transformers",
                   "We propose FastGF, a new efficient variant that improves GraphFormer on large graphs." + TAIL),
    "2201.00003": ("A benchmark of graph models",
                   "We introduce GBench, a new benchmark comparing graph neural networks and graph transformers." + TAIL),
}
BODY = {
    "2101.00002": "# Introduction\n\nGraphFormer [1] applies attention to graphs but is slow on large graphs. "
                  "Message passing networks [2] remain strong baselines.\n\n# References\n\n"
                  "1. A. Author. Attention is all you need for graphs. 2020.\n"
                  "2. B. Author. Neural message passing for quantum chemistry. 2017.\n"
                  "3. D. Author. Some unrelated older work on kernels. 2015.\n",
    "2201.00003": "# Introduction\n\nWe evaluate GraphFormer [1] and FastGF [2], two graph transformers. "
                  "GraphFormer [1] is now a standard component of many pipelines.\n\n# References\n\n"
                  "1. A. Author. Attention is all you need for graphs. 2020.\n"
                  "2. C. Author. Faster graph transformers. 2021.\n"
                  "3. D. Author. Some unrelated older work on kernels. 2015.\n",
}
DATES = {"2001.00001": "2020-01-10", "2101.00002": "2021-01-12", "2201.00003": "2022-01-15"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CS_CACHE", str(tmp_path / "cache"))
    from compilescholar.core import paths
    importlib.reload(paths)
    from compilescholar.dfc import store
    importlib.reload(store)
    from compilescholar.corpus import papers as PP
    importlib.reload(PP)
    con = PP.connect()
    for aid, (t, a) in ABS.items():
        con.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (aid, DATES[aid], t, PP.norm(t), a, json.dumps(["Author"]), "author", "cs.LG", "cs.LG", None, "test"))
    con.execute("INSERT INTO title_prefix SELECT substr(norm_title,1,40), arxiv_id FROM papers")
    con.commit()
    con.close()
    store.write_manifest("papers", {"test": True}, {"papers": 3})
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path / "data" / "external" / "ideaforecast"
    d.mkdir(parents=True)
    pq.write_table(pa.table({"arxiv_id": list(BODY), "month": ["x"] * 2, "title": ["t"] * 2, "text": list(BODY.values())}),
                   d / "2023-01.parquet")
    from compilescholar.documents import build as DB
    from compilescholar.citations import build as CB
    from compilescholar.extract import build as EB
    from compilescholar.extract import other_pass as O
    from compilescholar.extract import self_pass as S
    for m in (DB, CB, EB):
        importlib.reload(m)

    def fake_self(prompt, **k):
        if "method and experiment sections" in prompt:
            return json.dumps({"method": [], "findings": [], "limitations": [], "setting": {}})
        for aid, (t, a) in ABS.items():
            if a[:40] in prompt:
                name = a.split("We propose ")[-1].split(",")[0] if "propose" in a else "GBench"
                first = a.split(". ")[0].rstrip(".") + "."
                return json.dumps({"contributions": [{"text": first, "quote": first}],
                                   "proposes": [{"name": name, "aliases": [], "artefact": "method", "quote": first}],
                                   "findings": [{"text": s, "quote": s} for s in a.split(". ")[1:2]],
                                   "limitations": [], "setting": {}})
        return "{}"

    def fake_other(prompt, **k):
        import re as _re
        blocks = _re.split(r"\n\[(\d+)\] cited work:", prompt)[1:]
        pairs = []
        for i, block in zip(blocks[0::2], blocks[1::2]):
            i = int(i)
            comp = "standard component" in block
            pairs.append({"id": i, "function": "tool" if comp else "background", "relation": "uses" if comp else "background",
                          "about": "applies attention to graphs" if "applies attention" in block else None,
                          "facet": "method", "category": "graph transformers" if "graph transformers" in block else None,
                          "limitation": "slow on large graphs" if "slow on large graphs" in block else None})
        return json.dumps({"pairs": pairs})
    monkeypatch.setattr(S, "call_local", fake_self)
    monkeypatch.setattr(O, "call_local", fake_other)
    monkeypatch.setattr(S.run, "__defaults__", (None, fake_self, None))
    monkeypatch.setattr(O.run_batch, "__defaults__", (fake_other,))
    DB.build()
    CB.build()
    EB.build(categories=("cs.LG",), since="2018-01-01", n_deep=2, workers=2)
    return store


def test_quotes_are_substrings(env):
    con = env.connect("extract", readonly=True)
    cit = env.connect("citations", readonly=True)
    sents = {r[0] for r in cit.execute("SELECT sentence FROM sentences")}
    rows = list(con.execute("SELECT kind, quote, speaker, facet FROM statements"))
    assert {k for k, *_ in rows} == {"self", "other"}, list(con.execute("SELECT * FROM done_self"))
    for kind, quote, speaker, facet in rows:
        if kind == "other":
            assert quote in sents
        elif facet != "setting":
            assert " ".join(quote.split()).lower() in " ".join((ABS[speaker][1] + BODY.get(speaker, "")).split()).lower()
    tiers = dict(con.execute("SELECT arxiv_id, tier FROM done_self"))
    assert set(tiers.values()) == {"T1", "T2"}                    # deep tier only for papers with a full text


def test_as_of_monotone_and_ids(env):
    from compilescholar.cognition.asof import AsOf
    from compilescholar.cognition import profiles as P
    from compilescholar.tools import api
    early, late = AsOf("2021-06-30"), AsOf("2022-12-31")
    gf = "paper:2001.00001"
    e, l = P.profile(early, gf)["reception"], P.profile(late, gf)["reception"]
    assert e["n_citing"] == 1 and l["n_citing"] == 2                      # 2022 citer invisible in 2021
    assert all(s["date"] <= "2021-06-30" for s in early.statements())
    assert {s["id"] for s in early.statements()} <= {s["id"] for s in late.statements()}
    assert "slow on large graphs" in [x["text"] for x in l["limitations"]]
    assert AsOf("2020-06-30").paper("2101.00002") is None                  # not yet published
    api._Index._inst = None
    for b in api.closest_prior("graph attention transformers", "2021-06-30"):
        assert b["id"].startswith("stub:") or AsOf("2021-06-30").visible(b["id"])
        assert b["id"] != "paper:2201.00003"


def test_stale_upstream_blocks(env):
    from compilescholar.citations import build as CB
    from compilescholar.extract import build as EB
    env.write_manifest("papers", {"test": "changed"}, {"papers": 3})
    with pytest.raises(RuntimeError):
        EB.build()
    with pytest.raises(RuntimeError):
        CB.build()
