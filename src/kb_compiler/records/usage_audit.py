# -*- coding: utf-8 -*-
"""Governance: usage audit (the SUBTRACTION agenda, spec v1.1 治理对称条款).

Before every schema arbitration round, this runs and pairs with the gap list:
 - fill rate   : per field, fraction of records (of relevant kinds) with a
                 non-empty value
 - enum usage  : distribution of closed-vocabulary values (zero-use values are
                 retirement candidates — `cannot_tell` is the first on record)
 - view uses   : FIELD_USES counters from the view compiler (fields never
                 consumed downstream are dead weight in prompts + checks)
 - retirement candidates: low fill AND zero view use AND not required —
   reported, NEVER auto-removed (arbitration is human, same channel as adds)

Usage:
  python -m kb_compiler.records.usage_audit --records CHECKED.json \
      --views-stats VIEWS.json --out usage_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

from .schema import (ALL_KINDS, RELATIONS, RESULT_ROLES, CONFIG_ROLES,
                     ABSENCE_TYPES, EPISTEMIC, FINDING_STRENGTH,
                     FINDING_CLAIM_TYPES, LINEAGE_EVIDENCE, SHIFT_SOURCES,
                     REQUIRED_FIELDS, COMMON_FIELDS, DIMENSIONS)

ENUMS = {"relation": RELATIONS, "result.role": RESULT_ROLES, "config.role": CONFIG_ROLES,
         "absence_type": ABSENCE_TYPES, "epistemic": EPISTEMIC,
         "finding.strength": FINDING_STRENGTH, "finding.claim_type": FINDING_CLAIM_TYPES,
         "evidence_basis": LINEAGE_EVIDENCE, "source_type": SHIFT_SOURCES}

# Field names as STORED: slot.py resolves surface fields (method, scope_ref,
# target_ref, ...) into {surface, canonical, entity_id} dicts under <name>_ref
# and pops the raw field — auditing the raw names read only the empty-string
# residue and reported a false 0% fill (tool debt fix 09-07; that false zero
# masked the same mismatch in views.build_cards).
FIELDS_BY_KIND = {
    "result": ["method_ref", "measure", "role", "delta", "epistemic", "dims",
               "table_header", "quality_flag"],
    "config": ["method_ref", "item", "value", "applicability", "role",
               "epistemic", "quote"],
    "lineage": ["from_method_ref", "relation", "to_method_ref", "scope",
                "evidence_basis", "quote"],
    "finding": ["claim", "scope_ref_ref", "target_ref_ref", "condition",
                "strength", "claim_type", "quote"],
    "absence": ["subject", "missing", "absence_type", "evidence", "quote"],
    "shift": ["from_state", "to_state", "driver", "scope", "time_range",
              "source_type", "quote"],
    "notation": ["symbol", "quantity", "definition", "unit", "scope_ref_ref",
                 "quote"],
}

# Slice template explicitly allows empty ("无则空/可空/可选") — empty means
# inapplicable, a weak signal. Every other non-required per-kind field is
# slice-requested: low fill there is an EXTRACTION-MISS suspect, not dead
# weight (arbitration must distinguish — same discipline as the relation
# four-word zero-use case, 09-06).
SLICE_NULLABLE = {
    ("result", "delta"), ("result", "table_header"), ("result", "quality_flag"),
    ("finding", "scope_ref_ref"), ("finding", "target_ref_ref"),
    ("finding", "condition"), ("finding", "claim_type"),
    ("notation", "unit"), ("notation", "scope_ref_ref"),
    ("shift", "time_range"), ("lineage", "scope"), ("absence", "evidence"),
}


def _status(kind, field):
    if field in REQUIRED_FIELDS.get(kind, ()):
        return "required"
    if field in COMMON_FIELDS:
        return "common"
    if (kind, field) in SLICE_NULLABLE:
        return "slice_nullable"
    return "slice"


def audit(records_by_paper, views_field_uses=None):
    recs = []
    for pid, payload in records_by_paper.items():
        if pid == "canary":
            continue
        rs = payload.get("records", payload) if isinstance(payload, dict) else payload
        recs += rs
    by_kind = Counter(r.get("kind") for r in recs)
    fill = defaultdict(lambda: [0, 0])  # field -> [nonempty, applicable]
    enum_use = {name: Counter() for name in ENUMS}
    dims_use = Counter()
    for r in recs:
        kind = r.get("kind")
        # dims is the RESULT dims group; config/finding/lineage/shift/notation
        # carry their same-dimension groups under their own names
        # (applicability/condition/scope), audited per-kind below — counting
        # "dims" for them manufactures structural 0% noise candidates
        common = [f for f in COMMON_FIELDS if not (f == "dims" and kind != "result")]
        # dict.fromkeys: dedupe (quote sits in per-kind lists) so applicable
        # counts aren't double-inflated
        for f in dict.fromkeys(FIELDS_BY_KIND.get(kind, []) + common):
            v = r.get(f)
            fill[f"{kind}.{f}"][1] += 1
            if v not in (None, "", [], {}):
                fill[f"{kind}.{f}"][0] += 1
        for name, allowed in ENUMS.items():
            fld = name.split(".")[-1]
            if r.get("kind") == name.split(".")[0] or "." not in name:
                v = r.get(fld)
                if v:
                    enum_use[name][v] += 1
        for dgroup in ("dims", "condition", "scope", "applicability"):
            dg = r.get(dgroup) or {}
            if isinstance(dg, dict):
                for d in DIMENSIONS:
                    if dg.get(d):
                        dims_use[d] += 1
        if r.get("dims_new"):
            dims_use["dims_new(queue)"] += 1
    fill_rates = {k: {"nonempty": v[0], "applicable": v[1],
                      "rate": round(v[0] / v[1], 3) if v[1] else None,
                      "status": _status(k.split(".")[0], k.split(".", 1)[1])}
                  for k, v in sorted(fill.items())}
    enum_report = {}
    for name, allowed in ENUMS.items():
        used = enum_use[name]
        enum_report[name] = {"distribution": dict(used.most_common()),
                             "zero_use_values": [a for a in allowed if used.get(a, 0) == 0]}
    fu = (views_field_uses or {})
    candidates = []
    for k, v in fill_rates.items():
        field = k.split(".", 1)[1]
        if v["status"] == "required" or v["rate"] is None:
            continue
        if v["applicable"] < 10:
            continue  # statistically meaningless (e.g. notation n=3 in RL40)
        if v["rate"] < 0.02 and fu.get(field, 0) == 0:
            candidates.append({
                "field": k, "fill_rate": v["rate"], "view_uses": 0,
                "status": v["status"],
                "reason": ("low fill + zero use, but SLICE-REQUESTED: "
                           "extraction-miss suspect — verify before retiring"
                           if v["status"] == "slice"
                           else "low fill + zero downstream use")})
    for name, rep in enum_report.items():
        for z in rep["zero_use_values"]:
            candidates.append({"field": f"{name}={z}", "fill_rate": 0.0,
                               "reason": "enum value never used in corpus"})
    return {"n_records": len(recs), "by_kind": dict(by_kind),
            "fill_rates": fill_rates, "enum_usage": enum_report,
            "dims_usage": dict(dims_use), "view_field_uses": fu,
            "retirement_candidates": candidates,
            "note": "candidates are an AGENDA for human arbitration, never auto-applied"}


if __name__ == "__main__":
    from .common import load_json, save_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--views", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fu = {}
    if args.views:
        fu = (load_json(args.views, {}) or {}).get("stats", {}).get("field_uses", {})
    rep = audit(load_json(args.records, {}), fu)
    save_json(rep, args.out)
    print(f"records={rep['n_records']} kinds={rep['by_kind']}")
    print("retirement candidates:")
    for c in rep["retirement_candidates"]:
        print(" ", c)
