"""registry_dedup — fold duplicate entity rows created by the growth stage.

Background (G3 investigation, 2026-09-24): registry_growth's action=new path
appended a fresh entity whenever the LLM proposed a canonical, without
checking whether that canonical was already registered — a case/space
variant md5-collides with the existing entity_id and appends a duplicate
row. registry_v2.json carried 1,450 such rows (923 entity_ids appearing
2-11x). Consequences: views.cards keys by entity_id so later rows silently
overwrote earlier ones, and the answering-side grounding list showed the
same entity multiple times.

This pass is deterministic (no LLM): rows sharing an entity_id are merged
with unioned aliases/mention_papers, own-paper-authoritative entity_type,
and the richest non-null metadata. Source guard added in registry_growth.py
prevents recurrence; this tool repairs existing data.

Usage:
    python -m kb_compiler.records.registry_dedup \
        --registry kb/registry_v2.json --out kb/registry_v3.json \
        [--report kb/registry_dedup_report.json]
"""

from __future__ import annotations

import argparse

from .common import load_json, save_json


def _merge_group(rows: list[dict]) -> dict:
    """Merge duplicate rows of one entity_id into a single row.

    Preference rules (deterministic):
    - entity_id/canonical: first row (registry order = round-1 first)
    - aliases: ordered union, first row's order preserved
    - entity_type: row with in_corpus_paper_id wins (own-paper authority);
      else weighted majority by mention_count; tie -> first row
    - in_corpus_paper_id / origin_year_cited: first non-None
    - mention_papers: union; mention_count recomputed
    - provenance: unique values joined (round1 rows carry none)
    """
    out = dict(rows[0])
    aliases: list = []
    for r in rows:
        for a in r.get("aliases") or []:
            if a not in aliases:
                aliases.append(a)
    out["aliases"] = aliases
    own_rows = [r for r in rows if r.get("in_corpus_paper_id")]
    if own_rows:
        out["entity_type"] = own_rows[0]["entity_type"]
        out["in_corpus_paper_id"] = own_rows[0]["in_corpus_paper_id"]
    else:
        out["in_corpus_paper_id"] = None
        votes: dict[str, int] = {}
        for r in rows:
            votes[r.get("entity_type") or "method"] = (
                votes.get(r.get("entity_type") or "method", 0)
                + max(1, r.get("mention_count") or 1))
        best = max(votes.items(), key=lambda kv: (kv[1], -rows.index(
            next(r for r in rows if (r.get("entity_type") or "method") == kv[0]))))
        out["entity_type"] = best[0]
    years = [r.get("origin_year_cited") for r in rows
             if r.get("origin_year_cited")]
    out["origin_year_cited"] = years[0] if years else None
    papers: list = []
    for r in rows:
        for p in r.get("mention_papers") or []:
            if p not in papers:
                papers.append(p)
    out["mention_papers"] = sorted(papers)
    out["mention_count"] = len(papers)
    provs = []
    for r in rows:
        p = r.get("provenance")
        if p and p not in provs:
            provs.append(p)
    if provs:
        out["provenance"] = "+".join(provs)
    return out


def dedup_registry(registry: dict) -> tuple[dict, dict]:
    entities = registry.get("entities") or []
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for e in entities:
        eid = e["entity_id"]
        if eid not in groups:
            groups[eid] = []
            order.append(eid)
        groups[eid].append(e)
    merged = [dict(groups[eid][0]) if len(groups[eid]) == 1
              else _merge_group(groups[eid]) for eid in order]
    out = dict(registry)
    out["entities"] = merged
    n_dup_rows = sum(len(g) - 1 for g in groups.values() if len(g) > 1)
    report = {
        "entities_before": len(entities),
        "entities_after": len(merged),
        "duplicate_rows_folded": n_dup_rows,
        "entity_ids_affected": sum(1 for g in groups.values() if len(g) > 1),
    }
    return out, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()
    registry = load_json(args.registry, None)
    if registry is None:
        raise SystemExit(f"ERROR: cannot read registry {args.registry}")
    out, report = dedup_registry(registry)
    save_json(out, args.out)
    if args.report:
        save_json(report, args.report)
    print(f"registry dedup: {report['entities_before']} -> "
          f"{report['entities_after']} entities "
          f"({report['duplicate_rows_folded']} duplicate rows folded, "
          f"{report['entity_ids_affected']} entity_ids affected) -> {args.out}")


if __name__ == "__main__":
    main()
