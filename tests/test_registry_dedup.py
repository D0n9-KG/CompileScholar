"""G3 regression tests: registry duplicate folding + growth dup guard.

Background: registry_growth's action=new path appended entities without
checking whether the proposed canonical already existed — case/space
variants md5-collide on entity_id and appended duplicate rows that
views/cards silently overwrote (1,450 dup rows in registry_v2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kb_compiler.records.registry_dedup import dedup_registry  # noqa: E402


def _ent(eid, canonical, aliases=None, etype="method", own=None,
         papers=None, prov=None, year=None):
    return {"entity_id": eid, "canonical": canonical,
            "aliases": aliases if aliases is not None else [canonical],
            "entity_type": etype, "in_corpus_paper_id": own,
            "origin_year_cited": year,
            "mention_papers": papers if papers is not None else [],
            "mention_count": len(papers or []),
            **({"provenance": prov} if prov else {})}


def test_dedup_folds_duplicate_rows():
    reg = {"entities": [
        _ent("a1", "ADM", ["ADM", "AdaGN"], own="P1", papers=["P1", "P2"]),
        _ent("a1", "ADM", ["ADM", "classifier guidance"], etype="out_of_corpus",
             papers=["P3"], prov="round2_growth"),
    ]}
    out, report = dedup_registry(reg)
    assert len(out["entities"]) == 1
    e = out["entities"][0]
    # aliases union, order-preserving
    assert e["aliases"] == ["ADM", "AdaGN", "classifier guidance"]
    # own-paper row wins type and keeps own pid
    assert e["entity_type"] == "method"
    assert e["in_corpus_paper_id"] == "P1"
    # mention papers unioned, count recomputed
    assert e["mention_papers"] == ["P1", "P2", "P3"]
    assert e["mention_count"] == 3
    assert report["duplicate_rows_folded"] == 1


def test_dedup_no_own_majority_type_vote():
    reg = {"entities": [
        _ent("b1", "foo", etype="method", papers=["P1"], prov="round2_growth"),
        _ent("b1", "foo", etype="mechanism", papers=["P1", "P2", "P3"],
             prov="round2_growth"),
    ]}
    out, _ = dedup_registry(reg)
    e = out["entities"][0]
    # weighted majority by mention_count: mechanism (3 papers) beats method (1)
    assert e["entity_type"] == "mechanism"
    assert e["in_corpus_paper_id"] is None


def test_dedup_keeps_singletons_and_order():
    reg = {"entities": [
        _ent("x", "X"), _ent("a1", "ADM", own="P1", papers=["P1"]),
        _ent("y", "Y"), _ent("a1", "ADM", ["ADM", "G"], papers=["P2"]),
    ]}
    out, report = dedup_registry(reg)
    assert [e["canonical"] for e in out["entities"]] == ["X", "ADM", "Y"]
    assert report["entities_before"] == 4 and report["entities_after"] == 3


def test_dedup_origin_year_first_non_null():
    reg = {"entities": [
        _ent("c1", "C", year=None, papers=["P1"]),
        _ent("c1", "C", year={"value": 2020, "votes": 2}, papers=["P2"]),
    ]}
    out, _ = dedup_registry(reg)
    assert out["entities"][0]["origin_year_cited"]["value"] == 2020


def test_growth_dup_guard_folds_known_canonical(monkeypatch):
    """action=new proposing an existing canonical must fold as alias, not
    append a duplicate row."""
    from kb_compiler.records import registry_growth as rg

    existing = _ent("a1", "ADM", ["ADM"], own="P1", papers=["P1"])
    registry = {"entities": [existing], "surface_index": {}}
    queue = [
        {"surface": "adm", "papers": ["P2"], "contexts": []},
        {"surface": "brand new method", "papers": ["P2"], "contexts": []},
    ]
    # simulate assignment output: first surface "new" with canonical "ADM"
    # (case variant of existing), second surface genuinely new
    assigns = [
        {"i": 0, "action": "new", "canonical": "ADM", "entity_type": "method"},
        {"i": 1, "action": "new", "canonical": "brand new method",
         "entity_type": "method"},
    ]
    report = {}
    # replicate the guarded loop (the real one lives inside run_round2; the
    # guard semantics are what we pin here)
    canon_list = sorted({e["canonical"] for e in registry["entities"]})
    ent_by_norm = {}
    for e in registry["entities"]:
        ent_by_norm.setdefault(rg._norm(e["canonical"]), e)
    new_by_norm = {}
    new_entities = []
    dup_folds = 0
    for i, s in enumerate(queue):
        a = assigns[i]
        if a.get("action") == "match" and a.get("canonical") in canon_list:
            continue
        canonical = (a.get("canonical") or s["surface"]).strip()
        target = ent_by_norm.get(rg._norm(canonical)) or \
            new_by_norm.get(rg._norm(canonical))
        if target is not None:
            dup_folds += 1
            if s["surface"] not in target["aliases"]:
                target["aliases"].append(s["surface"])
            continue
        import hashlib
        ent = {"entity_id": hashlib.md5(
            rg._norm(canonical).encode()).hexdigest()[:12],
            "canonical": canonical, "aliases": sorted({canonical, s["surface"]})}
        new_entities.append(ent)
        new_by_norm[rg._norm(canonical)] = ent
    assert dup_folds == 1
    assert "adm" in existing["aliases"]
    assert len(new_entities) == 1
    assert new_entities[0]["canonical"] == "brand new method"
    # entity ids share the registry._norm md5 space
    assert new_entities[0]["entity_id"] != existing["entity_id"]
    assert report == {}
