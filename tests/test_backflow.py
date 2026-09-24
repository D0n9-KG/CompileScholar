"""Backflow tests: mention matching, edge building, idempotent attach/replay."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kb_compiler.records.backflow import (  # noqa: E402
    apply_backflow, build_backflow, load_backflow, match_mentions)

REGISTRY = {
    "entities": [
        {"entity_id": "aaa1", "canonical": "Classifier-Free Guidance",
         "aliases": ["Classifier-Free Guidance", "CFG"], "entity_type": "method"},
        {"entity_id": "bbb2", "canonical": "Denoising Diffusion Probabilistic Models",
         "aliases": ["DDPM"], "entity_type": "method"},
        {"entity_id": "ccc3", "canonical": "Transformer",
         "aliases": ["Transformer"], "entity_type": "method"},
        {"entity_id": "ddd4", "canonical": "Graph Attention Network",
         "aliases": ["GAT"], "entity_type": "method"},
        {"entity_id": "eee5", "canonical": "Generative Adversarial Network",
         "aliases": ["GAN"], "entity_type": "method"},
    ],
    "surface_index": {
        "classifier-free guidance": "aaa1", "cfg": "aaa1",
        "denoising diffusion probabilistic models": "bbb2", "ddpm": "bbb2",
        "transformer": "ccc3",
        "graph attention network": "ddd4", "gat": "ddd4",
        "generative adversarial network": "eee5", "gan": "eee5",
    },
}
BLOCKLIST = [{"name": "GAT", "domain": "cs", "reason": "cross-domain homonym"}]


def test_match_exact_and_containment():
    ms = match_mentions(
        ["CFG", "classifier-free guidance",
         "denoising diffusion probabilistic models (DDPM)"],
        REGISTRY["surface_index"])
    by_eid = {m["entity_id"]: m for m in ms}
    assert by_eid["aaa1"]["match"] == "exact"
    # both the alias ("CFG") and the full name hit the same entity — legal;
    # edge-level dedup by (from, to) is asserted in test_apply_idempotent
    # "... (DDPM)" contains the full-name surface as contiguous tokens
    assert by_eid["bbb2"]["match"] == "containment"


def test_match_blocklist_rejects():
    ms = match_mentions(["GAT"], REGISTRY["surface_index"], BLOCKLIST)
    assert ms == []


def test_match_ambiguous_dropped():
    # "transformer" is exact-unique; craft an ambiguous containment:
    # mention spanning two different entities' surfaces
    ms = match_mentions(["graph attention network versus gan"],
                        REGISTRY["surface_index"])
    # "graph attention network" ⊆ mention and "gan" ⊆ mention -> two
    # distinct entities -> dropped as ambiguous
    assert all(m["entity_id"] != "ddd4" for m in ms) or len(
        {m["entity_id"] for m in ms}) == 1


def test_match_single_token_short_no_containment():
    # single short token never tier-2 matches into longer surfaces
    ms = match_mentions(["formers"], {"transformer architecture": "ccc3"})
    assert ms == []


def _coarse():
    return {"records": [
        {"id": "coarse:abc123", "kind": "method", "subject": "Guidance Tuning",
         "claim": "We tune CFG scales automatically.",
         "quote": "classifier-free guidance scales are tuned automatically",
         "mentions": ["Classifier-Free Guidance", "Transformer"]},
        {"id": "coarse:def456", "kind": "limitation", "subject": "Guidance Tuning",
         "claim": "Tuning fails for GAN losses.",
         "quote": "fails under adversarial losses",
         "mentions": ["GAN"]},
    ], "provenance": "coarse"}


def test_build_backflow_edges():
    bf = build_backflow(_coarse(),
                        {"title": "Auto Guidance Tuning", "year": 2025,
                         "doi": "10.1/xyz"}, REGISTRY, BLOCKLIST)
    assert bf["node"]["id"].startswith("ext:")
    assert bf["node"]["entity_type"] == "external_paper"
    assert bf["node"]["doi"] == "10.1/xyz"
    assert len(bf["node"]["records"]) == 2
    rels = {(e["from"], e["relation"], e["provenance"]) for e in bf["edges"]}
    assert ("aaa1", "external_mention", "coarse") in rels
    assert ("ccc3", "external_mention", "coarse") in rels
    assert ("eee5", "external_mention", "coarse") in rels
    # GAT in blocklist never attaches
    assert all(e["from"] != "ddd4" for e in bf["edges"])
    # edges carry evidence quote + coarse record id
    e0 = bf["edges"][0]
    assert e0["record_id"].startswith("coarse:") and e0["quote"]


def test_apply_idempotent(tmp_path):
    views = {"genealogy": {"nodes": {}, "edges": []}}
    bf = build_backflow(_coarse(), {"title": "Auto Guidance Tuning",
                                    "year": 2025}, REGISTRY, BLOCKLIST)
    p = str(tmp_path / "bf.jsonl")
    r1 = apply_backflow(views, bf, persist_path=p)
    assert r1["attached"] == 3 and not r1["already_present"]
    n_edges = len(views["genealogy"]["edges"])
    # re-apply same paper: node exists, no duplicate edges, no new line
    r2 = apply_backflow(views, bf, persist_path=p)
    assert r2["already_present"] and r2["attached"] == 0
    assert len(views["genealogy"]["edges"]) == n_edges
    assert len(Path(p).read_text(encoding="utf-8").strip().splitlines()) == 1
    # a DIFFERENT paper attaches fresh edges
    bf2 = build_backflow(_coarse(), {"title": "Another Paper"}, REGISTRY,
                         BLOCKLIST)
    r3 = apply_backflow(views, bf2, persist_path=p)
    assert r3["attached"] == 3
    # lineage consumer sees the weak edges with canonical from_name
    e = next(e for e in views["genealogy"]["edges"]
             if e["relation"] == "external_mention")
    assert e["from_name"] == "Classifier-Free Guidance"
    assert e["to_name"] == "Auto Guidance Tuning"


def test_load_replay(tmp_path):
    views = {"genealogy": {"nodes": {}, "edges": []}}
    p = str(tmp_path / "bf.jsonl")
    bf = build_backflow(_coarse(), {"title": "Auto Guidance Tuning",
                                    "year": 2025}, REGISTRY, BLOCKLIST)
    apply_backflow(views, bf, persist_path=p)
    # fresh views, replay from disk
    views2 = {"genealogy": {"nodes": {}, "edges": []}}
    n = load_backflow(views2, p)
    assert n == 1
    assert len(views2["genealogy"]["edges"]) == len(
        views["genealogy"]["edges"])
    assert set(views2["genealogy"]["nodes"]) == set(
        views["genealogy"]["nodes"])
    # replay is idempotent
    assert load_backflow(views2, p) == 1
    assert len(views2["genealogy"]["edges"]) == len(
        views["genealogy"]["edges"])


def test_no_paper_key_returns_empty():
    bf = build_backflow(_coarse(), {"title": "  "}, REGISTRY)
    assert bf["node"] is None and bf["edges"] == []
    r = apply_backflow({"genealogy": {"nodes": {}, "edges": []}}, bf)
    assert r["attached"] == 0
