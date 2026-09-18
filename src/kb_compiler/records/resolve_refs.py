# -*- coding: utf-8 -*-
"""R1: single-pass end-of-build entity resolution (productionized relink).

Build-chain position (R1 merge, 2026-09-18 surgery batch 1):

  extract(slot) -> table channel -> merge -> round2 growth
  -> resolve_refs(FINAL registry)      <- this module, the ONLY authority
  -> views(same registry, resolved records)

Before R1, resolution ran at three sites against three registry versions:
slot-time binding (seed registry), views compile (another version), and the
ad-hoc relink_refs.py post-hoc pass (D2) — with version-skew disclosures.
After R1 there is exactly one authoritative pass, run after the registry
reaches its final state for the build; views consume resolved records, so
records and views can never disagree on registry version.

Rules (deterministic, order-independent — same as slot.py's own rule and
relink_refs.py's replay):
  1. null refs whose surface resolves in the final registry -> upgrade
     (canonical + entity_id), preserving F16 flags (f16_paper_scope,
     surface_raw) and every other field untouched.
  2. already-bound refs -> re-verify the surface still maps to the same
     entity_id. Drift is NEVER rewritten; it goes to the ledger as a warning
     for arbitration. Measured precedent (2026-09-18 equivalence run vs
     registry v37): 797 '+NEFT'-surface refs bound to the correct 'NEFT'
     entity at slot time, with surface_index later remapped to a junk
     duplicate '+NEFT' entity — auto-rewriting drift would have rebound all
     797 to the junk entity. Drift = governance signal, not a rewrite rule.
  3. unresolved surfaces -> counted, not enqueued (queue growth is slot's
     job at extraction time; by this point round2 already ran).

Usage:
  py -3.13 -m kb_compiler.records.resolve_refs --records REC.json \
      --registry REG.json --out OUT.json --ledger LEDGER.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys

REF_FIELDS = ("method_ref", "from_method_ref", "to_method_ref",
              "scope_ref_ref", "target_ref_ref")


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def resolve_records(data: dict, registry: dict):
    """Resolve every *_ref in {pid: payload} against registry. In-place.

    Returns (stats, ledger) where ledger is a list of change records
    ({paper_id, record_id, field, surface, old, new, kind}).
    """
    si = registry.get("surface_index", {})
    byid = {e["entity_id"]: e for e in registry.get("entities", [])}
    stats = {"upgraded": 0, "already_verified": 0, "still_null": 0,
             "drift_warning": 0, "no_surface": 0}
    ledger = []
    for pid, payload in data.items():
        if pid == "canary" or not isinstance(payload, dict):
            continue
        for r in payload.get("records", []):
            for field in REF_FIELDS:
                ref = r.get(field)
                if not isinstance(ref, dict):
                    continue
                surf = (ref.get("surface") or "").strip()
                if not surf:
                    stats["no_surface"] += 1
                    continue
                eid = si.get(norm(surf))
                if not eid or eid not in byid:
                    if ref.get("entity_id"):
                        # bound surface no longer resolves — governance change,
                        # flag loudly, never silently rewrite
                        stats["drift_warning"] += 1
                        ledger.append({
                            "paper_id": pid, "record_id": r.get("id"),
                            "field": field, "surface": surf, "kind": "drift",
                            "old": {"canonical": ref.get("canonical"),
                                    "entity_id": ref.get("entity_id")},
                            "new": None})
                    else:
                        stats["still_null"] += 1
                    continue
                if ref.get("entity_id") == eid:
                    stats["already_verified"] += 1
                    continue
                old = {"canonical": ref.get("canonical"),
                       "entity_id": ref.get("entity_id")}
                if old["entity_id"] is None:
                    # null -> bound: safe, deterministic upgrade
                    ref["canonical"] = byid[eid]["canonical"]
                    ref["entity_id"] = eid
                    stats["upgraded"] += 1
                    ledger.append({"paper_id": pid, "record_id": r.get("id"),
                                    "field": field, "surface": surf,
                                    "kind": "upgrade", "old": old,
                                    "new": {"canonical": ref["canonical"],
                                            "entity_id": eid}})
                else:
                    # bound -> different entity: drift, flag for arbitration,
                    # NEVER rewrite (old binding stands; see module docstring)
                    stats["drift_warning"] += 1
                    ledger.append({"paper_id": pid, "record_id": r.get("id"),
                                    "field": field, "surface": surf,
                                    "kind": "drift", "old": old,
                                    "new": {"canonical": byid[eid]["canonical"],
                                            "entity_id": eid}})
    return stats, ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = json.load(open(args.records, encoding="utf-8"))
    registry = json.load(open(args.registry, encoding="utf-8"))
    stats, ledger = resolve_records(data, registry)
    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(ledger, open(args.ledger, "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"resolve_refs: {json.dumps(stats)} changes={len(ledger)}")
    if stats["drift_warning"]:
        print(f"WARNING: {stats['drift_warning']} drift refs (bound surface "
              "maps differently in final registry) — inspect ledger; "
              "append-only growth should keep this at 0")


if __name__ == "__main__":
    main()
