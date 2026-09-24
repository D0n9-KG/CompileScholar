"""Tier-1 coarse extractor tests (prompt contract + record shape)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kb_compiler.records.coarse_extract import (  # noqa: E402
    _stable_id, extract_coarse)


def test_coarse_record_shape_via_parse():
    """extract_coarse's record-shaping logic against a mocked call_json."""
    import kb_compiler.records.coarse_extract as ce

    fake = {"records": [
        {"kind": "method", "subject": "CFG", "claim": "CFG trades off coverage and fidelity.",
         "quote": "classifier-free guidance can be used to trade off",
         "mentions": ["diffusion model", "classifier guidance"]},
        {"kind": "bogus_kind", "claim": "junk kind falls back to finding",
         "quote": "x"},
        {"claim": None},  # dropped: no claim
        {"kind": "limitation", "subject": "S", "claim": "Glycans overlooked.",
         "quote": "glycans... overlooked"},
    ]}
    orig = ce.call_json
    ce.call_json = lambda *a, **k: fake
    try:
        r = extract_coarse("Title", "A" * 200, 2024, "local:Qwen3.8-27B")
    finally:
        ce.call_json = orig
    recs = r["records"]
    assert len(recs) == 3
    assert r["provenance"] == "coarse"
    # kinds validated/fallback
    assert [x["kind"] for x in recs] == ["method", "finding", "limitation"]
    # coarse ids can never collide with Tier-2 hex ids
    assert all(x["id"].startswith("coarse:") for x in recs)
    # abstract claims are 'stated' by construction
    assert all(x["epistemic"] == "stated" for x in recs)
    # mention list bounded
    assert recs[0]["mentions"] == ["diffusion model", "classifier guidance"]


def test_coarse_short_abstract_skipped():
    r = extract_coarse("T", "too short", 2024, "local:Qwen3.8-27B")
    assert r["records"] == [] and "short" in r.get("note", "")


def test_stable_id_deterministic():
    a = _stable_id("title", "claim")
    assert a == _stable_id("title", "claim")
    assert a != _stable_id("title2", "claim")
    assert len(a) == 12
