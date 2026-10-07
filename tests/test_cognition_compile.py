# -*- coding: utf-8 -*-
"""Phase D④: the cognition READERS over the materialised tables (identity, lineage, families, facts, shifts,
comparisons, profiles) — as-of visibility, table-vs-statement sourcing, and graceful degradation before the
cognition stage exists. Built on the synthetic stage build of test_cognition_build (same fixture + LLM stubs)."""
from __future__ import annotations

import pytest

from test_cognition_build import PA, PB, PC, PD, _chat  # noqa: F401  (fixture constants + the LLM stub)
from test_cognition_build import env as cog_env          # noqa: F401  (pytest fixture re-use)


@pytest.fixture()
def store(cog_env):
    from compilescholar.cognition import build as CB
    CB.build(log=lambda *a: None, chat=_chat)
    return cog_env


def _view(T):
    from compilescholar.cognition.asof import AsOf
    return AsOf(T)


def test_identity_reads_materialised_adjudication(store):
    from compilescholar.cognition.identity import Identity
    ident = Identity(_view("2022-12-31"))
    assert ident.resolve("GraphNet") == PB and ident.resolve("graphnet") == PB    # deterministic 'ok'
    assert ident.resolve("FastGF") == PA
    assert ident.resolve("BiAttn") == PB                                          # LLM-adjudicated
    assert ident.status("biattn") == "adjudicated" and ident.status("graphnet") == "ok"
    assert ident.resolve("Transformer") is None and ident.status("transformer") is None
    assert "graphnet" in ident.aliases(PB)
    assert "GraphNet" in ident.surface_names(PB)          # from its own proposes statement


def test_lineage_valid_from_kinds_and_quotes(store):
    from compilescholar.cognition import lineage as L
    assert L.edges(_view("2021-12-31")) == []              # every edge's §2.4 valid_from is 2022+
    v = _view("2022-12-31")
    es = L.edges(v, PB)
    assert {(e["child"], e["parent"], e["relation"]) for e in es} == {(PD, PB, "extends"), (PD, PB, "improves")}
    ext = next(e for e in es if e["relation"] == "extends")
    assert ext["valid_from"] == "2022-01-01"
    assert len(ext["assertions"]) == 2 and {a["kind"] for a in ext["assertions"]} == {"self"}
    third = next(e for e in es if e["relation"] == "improves")
    assert third["assertions"][0]["kind"] == "third" and third["valid_from"] == "2022-06-01"
    qs = L.quotes(v, ext, k=1)
    assert qs and qs[0]["quote"] == "quote"                # fixture statements carry quote="quote"
    lo = L.lineage_of(v, PD)
    assert lo["parents"] and not lo["children"] and lo["combines"] == []
    assert L.edges(v, PC) == L.edges(v, PA) == []          # the merged id and the isolated paper have none


def test_families_snapshot_visibility(store):
    from compilescholar.cognition import families as F
    early = F.families(_view("2021-12-30"))
    assert early["snapshot"] is None and early["families"] == [] and early["before_grid"] is True
    got = F.families(_view("2022-03-15"))
    assert got["snapshot"] == "2022-03-01"                 # newest grid snapshot <= T, never clamped up
    fam = got["families"]
    assert len(fam) == 1 and sorted(fam[0]["members"]) == sorted([PB, PD])
    assert fam[0]["name"] == "graph transformer family"
    old = F.families(_view("2022-01-15"))["families"][0]
    assert old["named_by"] == "inherit"                    # only the latest snapshot is LLM-named
    v = _view("2022-12-31")
    assert F.family_of(v, PD)["family_id"] == F.families(v)["families"][0]["family_id"]
    assert F.family_of(v, PA) is None
    scoped = F.families(v, scope={PB})
    assert scoped["families"] == []                        # a 1-member remainder is below min_size


def test_facts_status_timeline_at_T(store):
    from compilescholar.cognition import facts as FA
    v20 = FA.facts(_view("2020-12-31"), members=[PB])
    # PB's own 2020 contribution statements are singleton facts; the limitations are 2021+
    assert len(v20) == 2 and {x["facet"] for x in v20} == {"contribution"}
    assert all(x["status"] == "single-source" and x["contested_since"] is None for x in v20)
    mid = FA.facts(_view("2021-06-30"), members=[PB])
    assert len(mid) == 3                                   # the limitation fact joins, contradiction not yet
    f = next(x for x in mid if x["facet"] == "limitation")
    assert f["status"] == "established"
    assert f["n_independent"] == 2 and f["n_subjects"] == 1 and f["contested_since"] is None
    assert {s["by"] for s in f["support"]} == {PA, PD} and f["first_seen"] == "2021-01-01"
    v = _view("2022-12-31")
    late = FA.facts(v, members=[PB])
    assert len(late) == 4                                  # the contradicting statement forms its own fact
    f2 = next(x for x in late if x["fact_id"] == f["fact_id"])
    assert f2["status"] == "established" and f2["contested_since"] == "2022-01-01"
    lims = [x for x in late if x["facet"] == "limitation"]
    assert len(lims) == 2 and all(x["contested_since"] == "2022-01-01" for x in lims)
    fam_facts = FA.facts(v, family_id=_family_id(v))       # + PD's three method statements (6, 7, 9), each
    assert {x["fact_id"] for x in late} < {x["fact_id"] for x in fam_facts}   # its own singleton fact
    assert len(fam_facts) == 7
    assert FA.facts(v, members=[]) == []


def _family_id(v):
    from compilescholar.cognition import families as F
    return F.families(v)["families"][0]["family_id"]


def test_shifts_approved_and_window_closed(store):
    from compilescholar.cognition import shifts as SH
    v = _view("2022-12-31")
    assert SH.events(v) == []                              # no fixture subject reaches MIN_N votes
    cog = store.connect("cognition")
    cog.execute("INSERT INTO shift_event(subject, facet, window_start, window_end, direction, evidence, status) "
                "VALUES (?,?,?,?,?,?,?)",
                (PB, "became_baseline", "2021-01-01", "2022-01-01",
                 '{"early": 0.1, "late": 0.6, "p": 0.01}', '["q"]', "approved"))
    cog.execute("INSERT INTO shift_event(subject, facet, window_start, window_end, direction, evidence, status) "
                "VALUES (?,?,?,?,?,?,?)",
                (PB, "became_component", "2022-01-01", "2022-06-01", "{}", "[]", "candidate"))
    cog.commit()
    cog.close()
    ev = SH.events(_view("2022-12-31"), PB)
    assert len(ev) == 1 and ev[0]["type"] == "became_baseline" and ev[0]["date"] == "2022-01-01"
    assert ev[0]["direction"]["late"] == 0.6 and ev[0]["evidence"] == ["q"]
    assert SH.events(_view("2021-12-31"), PB) == []        # the window has not closed at T
    assert SH.events(_view("2022-12-31"), PB, facet="became_component") == []   # candidates are not events


def test_comparisons_online_and_materialised(store):
    from compilescholar.cognition import comparisons as C
    v = _view("2022-12-31")
    g = C.compared_with(v, PA)
    assert g["n_compared_by"] == 1 and g["rows"][0]["by"] == PB
    assert g["rows"][0]["outcome"] == "citing_better" and g["outcomes"] == {"citing_better": 1}
    assert C.compared_with(_view("2019-12-31"), PA)["rows"] == []      # the citing sentence is 2020
    oe = C.outcome_edges(v, PA)
    assert len(oe) == 1 and oe[0]["citing"] == PB and oe[0]["cited"] == PA
    assert C.outcome_edges(_view("2019-12-31"), PA) == []
    assert C.baselines_in(v, [PB]) == [(PA, 1, {PB})]


def test_profile_mixes_tables_and_statements(store):
    from compilescholar.cognition import profiles as P
    pr = P.profile(_view("2022-12-31"), PA)
    assert pr["paper"]["title"] == "ArXiv title of A"
    r = pr["reception"]
    assert r["n_cites"] == 1 and r["first"] == r["last"] == "2020-06-01"        # reception_daily (materialised)
    assert r["n_citing"] == 1 and r["monthly"] == [{"month": "2020-06", "n": 1}]
    assert r["categories"] == [("graph attention models", 1)]                    # canonicalised phrase
    assert r["relation_share"] == {"compares": 1.0}
    assert len(pr["self"]) == 3 and any(s["name"] == "FastGF" for s in pr["self"])
    assert pr["shifts"] == [] and pr["evidence"] == {"n_cites": 1, "n_citing": 1, "months": 1, "has_self": True}
    early = P.profile(_view("2019-12-31"), PA)
    assert early["reception"]["n_cites"] == 0 and len(early["self"]) == 3        # self visible, no reception yet
    merged = P.profile(_view("2022-12-31"), PC)
    assert merged["object"] == PC and merged["paper"]["paper_id"] == PA          # canonical follows the alias


def test_readers_degrade_without_cognition(store):
    from pathlib import Path
    p = Path(str(store.db_path("cognition")))
    for suf in ("", "-wal", "-shm"):
        (p.parent / (p.name + suf)).unlink(missing_ok=True)
    v = _view("2022-12-31")
    assert v.cog is None
    from compilescholar.cognition import comparisons as C
    from compilescholar.cognition import families as F
    from compilescholar.cognition import facts as FA
    from compilescholar.cognition import lineage as L
    from compilescholar.cognition import profiles as P
    from compilescholar.cognition import shifts as SH
    from compilescholar.cognition.identity import Identity
    assert F.families(v)["families"] == [] and F.snapshot_at(v) == (None, None)
    assert SH.events(v) == [] and FA.facts(v, members=[PB]) == []
    assert L.edges(v) == [] and L.combines(v) == [] and C.outcome_edges(v) == []
    assert Identity(v).resolve("GraphNet") is None
    pr = P.profile(v, PA)                                   # the online half still works
    assert pr["reception"]["n_cites"] == 0 and len(pr["self"]) == 3
    assert C.compared_with(v, PA)["n_compared_by"] == 1     # statements-only reader is unaffected
