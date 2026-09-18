# -*- coding: utf-8 -*-
"""Unit tests for resolve_refs (R1). Run: py -3.13 -X utf8 test_resolve_refs.py"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from kb_compiler.records.resolve_refs import resolve_records  # noqa: E402


def make_registry():
    return {
        "entities": [
            {"entity_id": "E1", "canonical": "NEFT", "aliases": ["NEFT"],
             "entity_type": "method"},
            {"entity_id": "E2", "canonical": "+NEFT", "aliases": ["+NEFT"],
             "entity_type": "method"},
            {"entity_id": "E3", "canonical": "AdamW", "aliases": ["AdamW"],
             "entity_type": "method"},
        ],
        "surface_index": {"neft": "E1", "+neft": "E2", "adamw": "E3"},
    }


def run_case(name, records, registry, checks):
    stats, ledger = resolve_records(records, registry)
    for desc, fn in checks:
        assert fn(stats, ledger), f"[{name}] FAILED: {desc} (stats={stats})"
    print(f"[{name}] PASS  {stats}")
    return stats, ledger


def main():
    reg = make_registry()

    # 1. null upgrade: surface resolves -> bound, F16 flags preserved
    recs = {"p1": {"records": [
        {"id": "r1", "kind": "result",
         "method_ref": {"surface": "NEFT", "canonical": None,
                        "entity_id": None, "f16_paper_scope": True,
                        "surface_raw": "self"}}]}}
    run_case("null_upgrade_keeps_f16", recs, reg, [
        ("upgraded 1", lambda s, l: s["upgraded"] == 1),
        ("bound to E1", lambda s, l: recs["p1"]["records"][0]
            ["method_ref"]["entity_id"] == "E1"),
        ("f16 flag preserved", lambda s, l: recs["p1"]["records"][0]
            ["method_ref"]["f16_paper_scope"] is True),
        ("surface_raw preserved", lambda s, l: recs["p1"]["records"][0]
            ["method_ref"]["surface_raw"] == "self"),
    ])

    # 2. drift: bound to E1 but surface maps to E2 -> flagged, NOT rewritten
    recs = {"p1": {"records": [
        {"id": "r2", "kind": "result",
         "method_ref": {"surface": "+NEFT", "canonical": "NEFT",
                        "entity_id": "E1"}}]}}
    run_case("drift_not_rewritten", recs, reg, [
        ("drift flagged", lambda s, l: s["drift_warning"] == 1),
        ("old binding stands", lambda s, l: recs["p1"]["records"][0]
            ["method_ref"]["entity_id"] == "E1"),
        ("ledger records new target", lambda s, l:
            l[0]["new"]["entity_id"] == "E2" and l[0]["kind"] == "drift"),
    ])

    # 3. bound & consistent -> verified, no ledger entry
    recs = {"p1": {"records": [
        {"id": "r3", "kind": "config",
         "method_ref": {"surface": "AdamW", "canonical": "AdamW",
                        "entity_id": "E3"}}]}}
    run_case("already_verified", recs, reg, [
        ("verified", lambda s, l: s["already_verified"] == 1),
        ("no ledger", lambda s, l: len(l) == 0),
    ])

    # 4. null & unresolvable -> still_null, untouched
    recs = {"p1": {"records": [
        {"id": "r4", "kind": "finding",
         "scope_ref_ref": {"surface": "Mystery Method",
                           "canonical": None, "entity_id": None}}]}}
    run_case("still_null", recs, reg, [
        ("still_null", lambda s, l: s["still_null"] == 1),
        ("untouched", lambda s, l: recs["p1"]["records"][0]
            ["scope_ref_ref"]["entity_id"] is None),
    ])

    # 5. bound but surface no longer resolves anywhere -> drift flag, stands
    recs = {"p1": {"records": [
        {"id": "r5", "kind": "result",
         "method_ref": {"surface": "Retired Method", "canonical": "Old",
                        "entity_id": "E9"}}]}}
    run_case("bound_unresolvable", recs, reg, [
        ("drift flagged", lambda s, l: s["drift_warning"] == 1),
        ("binding stands", lambda s, l: recs["p1"]["records"][0]
            ["method_ref"]["entity_id"] == "E9"),
    ])

    # 6. canary + non-dict payloads skipped; all five ref fields covered
    recs = {"canary": {"records": [
        {"id": "c1", "method_ref": {"surface": "NEFT", "canonical": None,
                                    "entity_id": None}}]},
            "p2": "not-a-dict",
            "p3": {"records": [
                {"id": "r6", "from_method_ref": {"surface": "NEFT",
                                                 "canonical": None,
                                                 "entity_id": None},
                 "to_method_ref": {"surface": "nope", "canonical": None,
                                   "entity_id": None},
                 "target_ref_ref": "plain-string-skipped"}]}}
    run_case("canary_and_fields", recs, reg, [
        ("only p3 counted", lambda s, l: s["upgraded"] == 1
            and s["still_null"] == 1),
        ("canary untouched", lambda s, l: recs["canary"]["records"][0]
            ["method_ref"]["entity_id"] is None),
    ])

    print("ALL PASS")


if __name__ == "__main__":
    main()
