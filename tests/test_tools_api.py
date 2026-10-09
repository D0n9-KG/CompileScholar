# -*- coding: utf-8 -*-
"""Phase D④: the tools layer — quote/display separation, arm and ablation gating (server-side config the agent
cannot move), the budget ledger, as_of enforcement through the tools, external-hit merging with same-year date
resolution, and expand_citations over references + materialised cocite. Synthetic chain: the index fixture's
library + citations rows + a cognition build + an index build, fake embeddings, no network."""
from __future__ import annotations

import json

import pytest

from test_index_build import fake_embed
from test_index_build import env as idx_env           # noqa: F401  (fixture re-use)
from test_index_build import PA, PB, PD
from test_cognition_build import _chat

STUB = "stub:xray"


@pytest.fixture()
def tenv(idx_env, tmp_path, monkeypatch):
    IB, IS, store = idx_env
    cit = store.connect("citations")
    # sentence 1: PA (2019-06-01) cites PB + a stub together; sentence 2: PB (2020-06-01) cites PA
    cit.executemany("INSERT INTO cites VALUES (?,?,?,?,?,?,?,?)",
                    [(1, PA, "2019-06-01", 1, "b0", PB, 2, 0), (1, PA, "2019-06-01", 1, "b1", STUB, 2, 0),
                     (2, PB, "2020-06-01", 1, "b0", PA, 1, 1)])
    cit.executemany("INSERT INTO entries(citing, version, key, raw, title, year, doi, arxiv, cited, method) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(PA, 1, "b0", "raw b0", "Faster graph transformers", 2020, None, None, PB, "registry_title"),
                     (PA, 1, "b1", "raw b1", "X-ray", 2019, None, None, STUB, "stub"),
                     (PB, 1, "b0", "raw b0", "GraphFormer", 2019, None, None, PA, "registry_title")])
    cit.commit()
    cit.close()
    from compilescholar.cognition import build as CB
    CB.build(log=lambda *a: None, chat=_chat)
    IB.build(dense=("papers", "statements"), embed=fake_embed, model_label="fake")

    from compilescholar.tools import api
    api.reset_caches()                                    # a previous test's memo must not leak across stores
    api.set_index(IS.Index(embed=fake_embed, model_label="fake"))
    budget = tmp_path / "budget.jsonl"
    api.configure(api.ToolConfig(arm="full", budget_log=str(budget), external=False))
    yield type("Env", (), {"store": store, "api": api, "IS": IS, "budget": budget})()
    api.set_index(None)
    api.reset_caches()
    api.configure(api.ToolConfig())


def test_search_papers_quote_display_and_as_of(tenv):
    api = tenv.api
    got = api.search_papers("graphformer attention graphs", "2022-01-01", 5)
    assert got and got[0]["id"] == PA
    s = got[0]["self"]
    assert s["quote"] and s["display"] and len(s["display"]) <= 220
    early = api.search_papers("graphformer attention graphs", "2019-06-01", 5)
    assert PA in [x["id"] for x in early] and PB not in [x["id"] for x in early]
    assert all("field_says" in x for x in got)            # reception is on by default


def test_budget_ledger(tenv):
    api = tenv.api
    api.search_papers("graphs", "2022-01-01", 3)
    lines = [json.loads(x) for x in tenv.budget.read_text(encoding="utf-8").splitlines()]
    row = lines[-1]
    assert row["tool"] == "search_papers" and row["arm"] == "full"
    assert row["chars"] > 0 and row["tokens_est"] == max(1, row["chars"] // 4) and row["ms"] >= 0


def test_arms_gate_the_toolset(tenv):
    api = tenv.api
    assert set(api.active_tools()) >= {"search_papers", "field_map", "deep_read", "search_external"}
    api.configure(api.ToolConfig(arm="flat"))
    assert set(api.active_tools()) == {"search_flat"}
    got = api.search_flat("attention graphs", "2022-01-01", 4)
    assert set(got) == {"as_of", "passages"} and all("quote" in p for p in got["passages"])
    api.configure(api.ToolConfig(arm="none"))
    assert api.active_tools() == {}
    with pytest.raises(ValueError):
        api.configure(api.ToolConfig(arm="bogus"))
    with pytest.raises(ValueError):
        api.configure(api.ToolConfig(ablations=("nope",)))
    api.configure(api.ToolConfig(arm="full", external=False))


def test_ablation_reception(tenv):
    api = tenv.api
    api.configure(api.ToolConfig(arm="full", ablations=("-reception",), external=False))
    got = api.search_papers("graphformer attention graphs", "2022-01-01", 5)
    assert all("field_says" not in x and "n_cites" not in x for x in got)
    ev = api.find_evidence("beats graphs", "2022-01-01", 8)
    assert all(e["kind"] != "other" for e in ev)
    assert api.compared_with(PA, "2022-01-01")["ablated"] == "-reception"


def test_ablation_time_and_self_only(tenv):
    api = tenv.api
    api.configure(api.ToolConfig(arm="full", ablations=("-time",), external=False))
    assert api.frontier("graphs", "2022-01-01")["ablated"] == "-time"
    pr = api.paper_profile(PA, "2022-01-01")
    assert "shifts" not in pr and "monthly" not in pr["reception"]
    api.configure(api.ToolConfig(arm="full", ablations=("self_only",), external=False))
    assert api.field_map("graphs", "2022-01-01")["ablated"] == "self_only"
    ev = api.find_evidence("beats graphs", "2022-01-01", 8)
    assert all(e["kind"] in ("self", "passage") for e in ev)


def test_card_read_and_deep_read_visibility(tenv):
    api = tenv.api
    card = api.paper_card(PA, "2022-01-01")
    assert card["id"] == PA and card["title"] and isinstance(card["result_units"], list)
    assert all(("quote" in c and "display" in c) for c in card["contributions"])
    hits = api.read(PA, "2022-01-01", query="attention graphs", k=3)
    assert hits and all(h["quote"] and h["paper"] == PA for h in hits)
    early = api.read(PA, "2019-12-31", k=50)                    # section browse, no query
    assert early and all(h["date"] <= "2019-12-31" for h in early)
    late = api.read(PA, "2022-01-01", k=50)
    assert any(h["date"] == "2020-06-01" for h in late)          # the delta sentence appears at its date
    assert api.deep_read(PB, "2019-01-01")["error"] == "not visible at as_of"   # no LLM call on this path


def test_profile_and_field_tools_read_materialised(tenv):
    api = tenv.api
    pr = api.paper_profile(PA, "2022-01-01")
    assert pr["reception"]["n_cites"] == 1 and pr["reception"]["n_citing"] == 1
    assert pr["lineage"] == {"parents": [], "children": [], "combines": []}
    fm = api.field_map("graphs", "2022-01-01")
    assert fm["snapshot"] is None and fm["families"] == []       # this fixture builds no families at all
    assert fm["boundary"]["cited_not_resolved"] == 1              # stub:xray around the topic seeds
    miss = api.what_is_missing("graphs", "2022-01-01")
    assert any(u["id"] == STUB for u in miss["unresolved_cited"])


def test_expand_citations_local(tenv):
    api = tenv.api
    got = api.expand_citations([PB], "2022-01-01", 5)
    ids = {x["id"]: x for x in got["local"]}
    assert STUB in ids and ids[STUB]["n_cocite"] == 1 and "cocite" in ids[STUB]["via"]
    assert PA in ids and "references" in ids[PA]["via"]
    assert got["external"] == []                                  # external disabled in the fixture config


def test_search_external_merge_and_same_year(tenv, monkeypatch):
    api = tenv.api
    api.configure(api.ToolConfig(arm="full", external=True))
    from compilescholar.sources import refgraph, sciverse as SV

    def hit(title, year, doc="d1", chunk="verbatim chunk text"):
        return SV.SourceCandidate(source_name="sciverse-semantic", query_kind="semantic", status="ready",
                                  source_record_id=doc, title=title, year=year, venue="ICML",
                                  candidate_score=0.9, raw={"chunk": chunk, "doc_id": doc})

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def semantic_search(self, query, *, limit=10, year_lte=None):
            assert year_lte == 2021                              # as_of's year, same-year hits resolved per-hit
            return [hit("Faster graph transformers", 2020, "d1"),        # resolves to PB -> merged local
                    hit("Unknown same year work", 2021, "d2"),           # arxiv month BEFORE T -> external
                    hit("Unknown late work", 2021, "d3"),                # arxiv month AFTER T -> invisible
                    hit("Undated mystery", None, "d4"),                  # -> excluded (undated)
                    hit("Old external work", 2019, "d5")]                # -> external as-is

    lookups = {"Unknown same year work": {"arxiv": "2101.99999", "title": "Unknown same year work",
                                          "year": 2021, "month": 3, "abstract": ""},
               "Unknown late work": {"arxiv": "2112.99999", "title": "Unknown late work",
                                     "year": 2021, "month": 12, "abstract": ""}}
    monkeypatch.setattr(SV, "SciverseClient", FakeClient)
    monkeypatch.setattr(refgraph, "arxiv_lookup", lambda t: lookups.get(t))
    got = api.search_external("anything", "2021-06-30", 8)
    assert [x["id"] for x in got["local"]] == [PB]
    titles = [x["title"] for x in got["external"]]
    assert "Unknown same year work" in titles and "Old external work" in titles
    assert "Unknown late work" not in titles and "Undated mystery" not in titles
    same_year = next(x for x in got["external"] if x["title"] == "Unknown same year work")
    assert same_year["date"] == "2021-03-31"                      # month precision, visible at its hi
    assert got["excluded_same_year_undated"] == 0 and got["excluded_undated"] == 1
    ext = next(x for x in got["external"] if x["title"] == "Old external work")
    assert ext["quote"] == "verbatim chunk text" and ext["external"] is True


def test_summary_arm_compresses(tenv, monkeypatch):
    api = tenv.api
    from compilescholar.llm import client as LC
    calls = []

    def fake_call_local(prompt, **kw):
        calls.append(prompt)
        return json.dumps({"compressed": True, "kept": "ids and quotes"})

    monkeypatch.setattr(LC, "call_local", fake_call_local)
    api.configure(api.ToolConfig(arm="summary", external=False))
    got = api.search_papers("graphs", "2022-01-01", 3)
    assert got == {"compressed": True, "kept": "ids and quotes"} and calls


def test_family_neighbors_shape_and_before_grid(tenv):
    api = tenv.api
    got = api.family_neighbors([PA, PB], "2022-01-01", 8)
    assert got["seeds"] == [PA, PB] and isinstance(got["local"], list)
    for r in got["local"]:
        assert {"id", "n_shared_seeds", "family", "family_id", "n_members"} <= set(r)
        assert r["id"] not in (PA, PB) and r["n_shared_seeds"] >= 1
    early = api.family_neighbors([PA], "1900-01-01", 8)      # before the snapshot grid
    assert early["local"] == [] and "before_grid" in early
    assert api.family_neighbors(PA, "2022-01-01", 8)["seeds"] == [PA]   # a bare string seed is accepted
