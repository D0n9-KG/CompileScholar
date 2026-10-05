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
            base = "strong baselines" in block
            fn = "tool" if comp else "baseline" if base else "background"
            pairs.append({"id": i, "function": fn, "relation": "uses" if comp else "compares" if base else "background",
                          "about": "applies attention to graphs" if "applies attention" in block else None,
                          "facet": "method", "category": "graph transformers" if "graph transformers" in block else None,
                          "limitation": "slow on large graphs" if "slow on large graphs" in block else None,
                          "name": "GraphFormer" if "GraphFormer" in block.split("sentence:")[-1] and
                                  "Attention is all you need" in block else None,
                          "outcome": None, "builds_on": None, "builds_on_relation": None})
        return json.dumps({"pairs": pairs})
    monkeypatch.setattr(S, "call_local", fake_self)
    monkeypatch.setattr(O, "call_local", fake_other)
    monkeypatch.setattr(S.run, "__defaults__", (None, fake_self, None))
    monkeypatch.setattr(O.run_batch, "__defaults__", (fake_other,))
    DB.build()
    CB.build()
    EB.build(categories=("cs.LG",), since="2018-01-01", n_deep=2, workers=2)
    from compilescholar.index import build as IB
    importlib.reload(IB)
    IB.build()
    from compilescholar.tools import api
    importlib.reload(api)
    api.set_index(IB.Index(embed=None))          # BM25-only in tests
    return store


def test_quotes_are_substrings(env):
    con = env.connect("extract", readonly=True)
    cit = env.connect("citations", readonly=True)
    sents = {r[0] for r in cit.execute("SELECT sentence FROM sentences")}
    rows = list(con.execute("SELECT kind, quote, speaker, facet FROM statements"))
    assert {k for k, *_ in rows} == {"self", "other"}, list(con.execute("SELECT * FROM tiers"))
    for kind, quote, speaker, facet in rows:
        if kind == "other":
            assert quote in sents
        elif facet != "setting":
            assert " ".join(quote.split()).lower() in " ".join((ABS[speaker][1] + BODY.get(speaker, "")).split()).lower()
    tiers = dict(con.execute("SELECT arxiv_id, tier FROM tiers"))
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
    for b in api.closest_prior("graph attention transformers", "2021-06-30"):
        assert b["id"].startswith("stub:") or AsOf("2021-06-30").visible(b["id"])
        assert b["id"] != "paper:2201.00003"


def _ids(x):
    """Every 'paper:...' id anywhere in a tool result."""
    if isinstance(x, dict):
        return [i for v in x.values() for i in _ids(v)]
    if isinstance(x, list):
        return [i for v in x for i in _ids(v)]
    return [x] if isinstance(x, str) and x.startswith("paper:") else []


def test_every_tool_respects_as_of(env):
    from compilescholar.cognition.asof import AsOf
    from compilescholar.tools import api
    T = "2021-06-30"
    view = AsOf(T)
    calls = {
        "search_papers": ("graph attention",), "paper_card": ("2001.00001",), "read": ("2001.00001",),
        "find_evidence": ("attention to graphs",), "field_map": ("graph transformers",),
        "paper_profile": ("2001.00001",), "closest_prior": ("attention graph model",),
        "baselines_for": ("graph models",), "open_issues": ("graph transformers",), "frontier": ("graph",),
        "compared_with": ("2001.00001",), "what_is_missing": ("graph",), "citations_of": ("2001.00001",),
        "references_of": ("2101.00002",),
    }
    assert set(calls) == set(api.TOOLS)
    for name, args in calls.items():
        out = api.TOOLS[name](*args, T)
        for i in _ids(out):
            assert view.visible(i), (name, i)
    card = api.paper_card("2101.00002", "2022-12-31")
    assert card["proposes"] and card["proposes"][0]["name"] == "FastGF"
    assert api.paper_card("2201.00003", T).get("error")                    # published after T
    assert api.citations_of("2001.00001", T) == [{"id": "paper:2101.00002", "date": "2021-01-12",
                                                    "title": "Faster graph transformers"}]
    late = api.compared_with("2001.00001", "2022-12-31")
    assert late["n_compared_by"] == 0 or all(r["by"] != "paper:2201.00003" or r["date"] <= "2022-12-31"
                                             for r in late["rows"])


def test_answer_lit_mode_uses_tools_and_verbatim_snippets(env, monkeypatch):
    from compilescholar.answer import pipeline as AP
    from compilescholar.tools import api

    def fake_chat(prompt, max_tokens=6000, temperature=0.2):
        if '"sections"' in prompt:
            return json.dumps({"sections": [{"title": "Graph transformers", "goal": "what exists",
                                             "queries": ["graph attention model"]}]})
        if "off_topic" in prompt:
            return json.dumps({"off_topic": []})
        ids = re_ids.findall(prompt)
        return f"GraphFormer applies attention to graph data [{ids[0]}]." if ids else ""
    import re as _re
    re_ids = _re.compile(r"\[(E\d+)\]")
    monkeypatch.setattr(AP, "chat", fake_chat)
    r = AP.answer_lit("What graph attention models exist?", cutoff="2021-07", tools=api)
    assert r["trace"]["as_of"] == "2021-06-30" and r["trace"]["mode"] == "lit"
    secs = r["sections"]
    assert secs and secs[0]["citations"]
    corpus = " ".join(a for _, a in ABS.values()) + " ".join(BODY.values())
    for c in secs[0]["citations"]:
        for sn in c["snippets"]:
            assert " ".join(sn.split()) in " ".join(corpus.split())    # verbatim source text, never system text
        assert c["title"] != "A benchmark of graph models"                # 2022 paper invisible at 2021-06-30


def test_stale_upstream_blocks(env):
    from compilescholar.citations import build as CB
    from compilescholar.extract import build as EB
    env.write_manifest("papers", {"test": "changed"}, {"papers": 3})
    with pytest.raises(RuntimeError):
        EB.build()
    with pytest.raises(RuntimeError):
        CB.build()


def test_rebuild_after_no_change_does_nothing_and_llm_failure_is_not_done(env, monkeypatch):
    """Probe P5 on the real stages: a dead LLM leaves the items open (not done, no statements, manifest incomplete),
    a second build retries them; a build with nothing changed calls no LLM at all."""
    from compilescholar.extract import build as EB
    from compilescholar.extract import other_pass as O
    from compilescholar.extract import self_pass as S
    con = env.connect("extract", readonly=True)
    before = con.execute("SELECT count(*) FROM statements").fetchone()[0]
    before_other = con.execute("SELECT count(*) FROM statements WHERE pass='other'").fetchone()[0]
    con.close()
    alive = S.run.__defaults__[1]
    calls = []

    def dead(prompt, **k):
        calls.append(1)
        return None

    monkeypatch.setattr(S.run, "__defaults__", (None, dead, None))
    monkeypatch.setattr(O.run_batch, "__defaults__", (dead,))
    EB.build(categories=("cs.LG",), since="2018-01-01", n_deep=2, workers=2)
    assert calls == []                                    # nothing changed -> no work
    m = env.read_manifest("extract")
    assert m["complete"] and m["work"].get("ok")

    # a changed self prompt re-opens the self pass only; with a dead server every item fails, none is done
    monkeypatch.setattr(S, "PROMPT_SHA", "changed-prompt")
    EB.build(categories=("cs.LG",), since="2018-01-01", n_deep=2, workers=2)
    assert len(calls) == 3                                # 3 self items, 0 other items
    m = env.read_manifest("extract")
    assert not m["complete"] and m["work"].get("failed") == 3
    con = env.connect("extract", readonly=True)
    # a failed item has no output under its current key (same as a clean build): its old-prompt rows are gone,
    # the other pass is untouched
    assert con.execute("SELECT count(*) FROM statements WHERE pass='self'").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM statements WHERE pass='other'").fetchone()[0] == before_other
    assert {r[0] for r in con.execute("SELECT status FROM _work WHERE pass='self'")} == {"failed"}
    con.close()
    # the server comes back: the next build redoes exactly the failed items
    monkeypatch.setattr(S.run, "__defaults__", (None, alive, None))
    EB.build(categories=("cs.LG",), since="2018-01-01", n_deep=2, workers=2)
    m = env.read_manifest("extract")
    assert m["complete"]
    con = env.connect("extract", readonly=True)
    assert con.execute("SELECT count(*) FROM statements").fetchone()[0] == before
    con.close()
