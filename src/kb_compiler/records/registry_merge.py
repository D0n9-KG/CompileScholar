# -*- coding: utf-8 -*-
"""B8 first cut: deterministic registry duplicate merge (zero LLM).

Measured debt (registry_v38_d2, 2026-09-18): 4 '+X'-duplicate-of-X entities
('+NEFT' etc. — the root cause of 797 drift refs found by resolve_refs) and
187 exact canonical duplicates (normalized: 'noisynet' x4, 'qr-dqn' x2,
plus junk labels). Total 6253 entities.

Merge policy (deterministic, fully audited):
  survivor = entity with the most record refs; tie-break: more surface_index
  entries, then more aliases, then lexicographically smallest entity_id.
  merged-away entity: aliases unioned into survivor, mention_papers unioned,
  surface_index entries remapped, entity dropped.

This is a MERGE (intentional governance action), not drift (unexpected) —
so record refs ARE rebound to the survivor, every change ledgered. Gray-zone
categories (generic labels, ablation variants, junk LaTeX...) are NOT touched
here — those belong to the LLM-arbitration pass with a human spot-check gate.

Usage:
  py -3.13 -m kb_compiler.records.registry_merge --registry REG.json \
      --records REC.json --out OUT.json --ledger LEDGER.json [--apply-records REC_OUT]
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


def find_merge_groups(entities):
    """Equivalence classes of entities that are deterministically the same.

    Union rules (zero LLM):
      (a) same entity_id (measured: 187 duplicate-id groups in v38 — growth
          rounds appended same-id rows with different alias sets);
      (b) same normalized canonical;
      (c) '+X' canonical and plain 'X' both exist ('+X' merges into 'X').
    Union-find over entity indices — no chain-resolution cycles possible
    (the 2026-09-18 chain version infinite-looped on a self-pair when two
    rows shared an entity_id).

    Returns {away_index: keep_index} collapsed to ROOT classes.
    """
    n = len(entities)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_id = {}
    by_norm = {}
    for i, e in enumerate(entities):
        by_id.setdefault(e.get("entity_id"), []).append(i)
        by_norm.setdefault(norm(e.get("canonical")), []).append(i)
    for idxs in by_id.values():
        for j in idxs[1:]:
            union(idxs[0], j)
    for idxs in by_norm.values():
        for j in idxs[1:]:
            union(idxs[0], j)
    for c, idxs in by_norm.items():
        if c.startswith("+"):
            base = by_norm.get(c[1:].strip()) or []
            if base:
                for i in idxs:
                    union(base[0], i)
    classes = {}
    for i in range(n):
        r = find(i)
        if r != i:
            classes[i] = r
    return classes


def elect_survivors(entities, groups, ref_counts, si_counts, prefer_idx=None):
    """Per class: survivor = most refs; tie-break si entries, aliases,
    entity_id. prefer_idx: indices that win ties absolutely (plain-name
    entities over '+X' duplicates). Returns {away_entity_id: survivor_entity_id}.
    NOTE: superseded by the inline index-level election in merge_registry
    (same-id row duplicates make id-level mapping ambiguous); kept for
    external/test use on id-clean registries."""
    from collections import defaultdict
    members = defaultdict(set)
    for away_i, keep_i in groups.items():
        members[keep_i].add(away_i)

    def score(i):
        e = entities[i]
        return (0 if prefer_idx and i in prefer_idx else 1,
                -ref_counts.get(e["entity_id"], 0),
                -si_counts.get(e["entity_id"], 0),
                -len(e.get("aliases") or []), e["entity_id"])

    elected = {}
    for keep_i, aways in members.items():
        pool = sorted([keep_i] + list(aways), key=score)
        survivor = pool[0]
        for i in pool:
            if i != survivor:
                elected[entities[i]["entity_id"]] = entities[survivor]["entity_id"]
    return elected


def merge_registry(registry, records=None):
    entities = registry.get("entities", [])
    ref_counts = {}
    if records:
        for pid, payload in records.items():
            if pid == "canary" or not isinstance(payload, dict):
                continue
            for r in payload.get("records", []):
                for f in REF_FIELDS:
                    ref = r.get(f)
                    if isinstance(ref, dict) and ref.get("entity_id"):
                        ref_counts[ref["entity_id"]] = \
                            ref_counts.get(ref["entity_id"], 0) + 1
    si_counts = {}
    for v in registry.get("surface_index", {}).values():
        si_counts[v] = si_counts.get(v, 0) + 1

    groups = find_merge_groups(entities)
    # plus-dup discipline: the PLAIN entity ('X') always wins over its '+X'
    # duplicate (the '+NEFT' junk held 18 refs from slot-time binding — refs
    # then rebind to the survivor; the canonical name must still be 'NEFT')
    prefer_idx = {i for i in range(len(entities))
                  if not (entities[i].get("canonical") or "").startswith("+")}

    # classes at INDEX level (rows, not ids — the registry carries same-id
    # rows with different alias sets; id-level bookkeeping would drop the
    # survivor together with its duplicates)
    from collections import defaultdict
    members = defaultdict(set)
    for away_i, keep_i in groups.items():
        members[keep_i].add(away_i)

    def score(i):
        e = entities[i]
        return (0 if i in prefer_idx else 1,
                -ref_counts.get(e["entity_id"], 0),
                -si_counts.get(e["entity_id"], 0),
                -len(e.get("aliases") or []), e["entity_id"])

    ledger = []
    drop_rows = set()          # away INDICES to remove
    id_remap = {}              # away entity_id -> survivor entity_id (cross-id only)
    for keep_i, aways in members.items():
        pool = sorted([keep_i] + list(aways), key=score)
        surv = pool[0]
        survivor = entities[surv]
        survivor.setdefault("aliases", [])
        for i in pool:
            if i == surv:
                continue
            away = entities[i]
            for a in [away.get("canonical")] + (away.get("aliases") or []):
                if a and a not in survivor["aliases"] \
                        and a != survivor.get("canonical"):
                    survivor["aliases"].append(a)
            survivor["mention_papers"] = sorted(
                set(survivor.get("mention_papers") or []) |
                set(away.get("mention_papers") or []))
            if away.get("in_corpus_paper_id") and \
                    not survivor.get("in_corpus_paper_id"):
                survivor["in_corpus_paper_id"] = away.get("in_corpus_paper_id")
            drop_rows.add(i)
            same_id = away["entity_id"] == survivor["entity_id"]
            if not same_id:
                id_remap[away["entity_id"]] = survivor["entity_id"]
            ledger.append({
                "away": {"row": i, "entity_id": away["entity_id"],
                         "canonical": away.get("canonical")},
                "keep": {"row": surv, "entity_id": survivor["entity_id"],
                         "canonical": survivor.get("canonical")},
                "reason": ("same_id_row_dup" if same_id else
                           "plus_dup" if
                           norm(away.get("canonical")).startswith("+") and
                           norm(away.get("canonical"))[1:].strip() ==
                           norm(survivor.get("canonical"))
                           else "exact_norm_dup")})

    registry["entities"] = [e for i, e in enumerate(entities)
                            if i not in drop_rows]
    n_si = 0
    for k, v in registry.get("surface_index", {}).items():
        if v in id_remap:
            registry["surface_index"][k] = id_remap[v]
            n_si += 1

    # post-condition: no duplicate entity_ids remain
    seen = set()
    for e in registry["entities"]:
        assert e["entity_id"] not in seen, \
            f"post-merge duplicate id: {e['entity_id']}"
        seen.add(e["entity_id"])

    n_refs = 0
    if records:
        canon_of = {e["entity_id"]: e["canonical"]
                   for e in registry["entities"]}
        for pid, payload in records.items():
            if pid == "canary" or not isinstance(payload, dict):
                continue
            for r in payload.get("records", []):
                for f in REF_FIELDS:
                    ref = r.get(f)
                    if isinstance(ref, dict) and \
                            ref.get("entity_id") in id_remap:
                        old = {"canonical": ref.get("canonical"),
                               "entity_id": ref["entity_id"]}
                        ref["entity_id"] = id_remap[old["entity_id"]]
                        ref["canonical"] = canon_of[ref["entity_id"]]
                        n_refs += 1
                        ledger.append({"record": {"paper_id": pid,
                                                  "record_id": r.get("id"),
                                                  "field": f},
                                       "old": old,
                                       "new": {"canonical": ref["canonical"],
                                               "entity_id": ref["entity_id"]},
                                       "reason": "merge_rebind"})
    stats = {"entities_before": len(entities),
             "entities_after": len(registry["entities"]),
             "rows_merged_away": len(drop_rows), "si_remapped": n_si,
             "refs_rebound": n_refs}
    return registry, records, stats, ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--records", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--apply-records", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    registry = json.load(open(args.registry, encoding="utf-8"))
    records = json.load(open(args.records, encoding="utf-8")) \
        if args.records else None
    registry, records, stats, ledger = merge_registry(registry, records)
    json.dump(registry, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump(ledger, open(args.ledger, "w", encoding="utf-8"),
              ensure_ascii=False)
    if records is not None and args.apply_records:
        json.dump(records, open(args.apply_records, "w", encoding="utf-8"),
                  ensure_ascii=False)
        print(f"records with merged refs -> {args.apply_records}")
    print(f"merge: {json.dumps(stats)} ledger={len(ledger)}")


if __name__ == "__main__":
    main()
