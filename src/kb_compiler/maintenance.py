# -*- coding: utf-8 -*-
"""Block 5: invalidation algebra (failure algebra, 失效代数).

Four primitives per blueprint:
 1. provenance index   : record_id -> every view location consuming it
                         (views carry record_id in cells/edges/cards by design —
                         this is the payoff of that discipline)
 2. row-level invalidation: drop records (by id or whole paper) -> views are a
                         PURE FUNCTION of records, so invalidation = filter +
                         recompile affected; delta report quantifies blast radius
 3. dream queue (做梦式重算): papers needing re-extraction — invalidated, or
                         source-drifted (text sha256 != manifest.source_version)
 4. two-phase commit   : compile to .staging -> validate -> atomic os.replace;
                         a failed validation never corrupts the live views
"""
from __future__ import annotations

import hashlib
import json
import os
import sys


# ---------- 1. provenance index ----------

def build_provenance_index(views: dict) -> dict:
    """record_id -> list of {"view": name, "loc": descriptor}."""
    idx = {}

    def _add(rid, view, loc):
        if rid:
            idx.setdefault(rid, []).append({"view": view, "loc": loc})

    for key, ents in views.get("matrix", {}).get("tables", {}).items():
        for ent, cells in ents.items():
            for c in cells:
                _add(c.get("record_id"), "matrix", f"{key}|{ent}")
    for e in views.get("genealogy", {}).get("edges", []):
        _add(e.get("record_id"), "genealogy", f"{e.get('from_name')}->{e.get('to_name')}")
    for a in views.get("coverage", {}).get("absences_extracted", []):
        _add(a.get("record_id"), "coverage", a.get("subject"))
    for card in views.get("cards", {}).get("cards", {}).values():
        for sect in ("configs", "main_results", "ablations", "findings",
                     "lineage_out", "deltas_explicit"):
            for x in card.get(sect, []):
                _add(x.get("record_id"), "cards", f"{card.get('canonical')}|{sect}")
    for s in views.get("narrative", {}).get("shifts", []):
        _add(s.get("record_id"), "narrative", s.get("scope"))
    for ln in views.get("narrative", {}).get("family_lines", []):
        for st in ln.get("steps", []):
            _add(st.get("record_id"), "narrative", ln.get("theme"))
    for d in views.get("pair_deltas", []):
        _add(d.get("leader_record"), "pair_deltas", d.get("subject"))
        _add(d.get("follower_record"), "pair_deltas", d.get("subject"))
    return idx


# ---------- 2+3. invalidation + dream queue ----------

def invalidate(records_by_paper: dict, paper_ids=(), record_ids=()):
    """Returns (filtered_records, report). Row-level: record_ids; paper-level:
    paper_ids (source retracted / new version / corrupted extraction)."""
    rid_set = set(record_ids)
    out, removed = {}, {"by_paper": {}, "record_ids": []}
    for pid, payload in records_by_paper.items():
        recs = payload.get("records", []) if isinstance(payload, dict) else payload
        keep = []
        n_rm = 0
        for r in recs:
            if pid in paper_ids or r.get("id") in rid_set:
                n_rm += 1
                removed["record_ids"].append(r.get("id"))
            else:
                keep.append(r)
        if pid in paper_ids:
            removed["by_paper"][pid] = n_rm
            continue  # whole paper drops out
        if isinstance(payload, dict):
            out[pid] = {**payload, "records": keep}
        else:
            out[pid] = keep
    removed["n"] = len(removed["record_ids"])
    return out, removed


def source_drift(texts_dir: str, manifest: dict) -> list:
    """sha256(current text)[:16] vs manifest.source_version -> stale papers."""
    stale = []
    for pid, m in manifest.items():
        sf = m.get("source_file")
        if not sf or not os.path.exists(sf):
            continue
        h = hashlib.sha256(open(sf, "rb").read()).hexdigest()[:16]
        if h != m.get("source_version"):
            stale.append({"paper_id": pid, "manifest_hash": m.get("source_version"),
                          "current_hash": h})
    return stale


def dream_queue(invalidation_report: dict, drift: list) -> list:
    """papers awaiting re-extraction (做梦式重算 batch input)."""
    q = [{"paper_id": p, "reason": "invalidated"} for p in invalidation_report.get("by_paper", {})]
    q += [{"paper_id": d["paper_id"], "reason": "source_drift"} for d in drift]
    return q


# ---------- 4. two-phase commit ----------

def validate_views(views: dict) -> list:
    errs = []
    stats = views.get("stats") or {}
    if stats.get("n_records", 0) <= 0:
        errs.append("empty record set")
    for k in ("matrix", "genealogy", "coverage", "cards", "narrative"):
        if k not in views:
            errs.append(f"missing view: {k}")
    # provenance discipline: matrix cells must carry record_id+paper_id
    for key, ents in views.get("matrix", {}).get("tables", {}).items():
        for ent, cells in ents.items():
            for c in cells:
                if not c.get("record_id") or not c.get("paper_id"):
                    errs.append(f"cell without provenance: {key}|{ent}")
                    break
    return errs


def commit_views(views: dict, live_path: str) -> dict:
    """staging -> validate -> atomic replace. Never corrupts live on failure."""
    errs = validate_views(views)
    if errs:
        return {"committed": False, "errors": errs[:20]}
    staging = live_path + ".staging"
    with open(staging, "w", encoding="utf-8") as f:
        json.dump(views, f, ensure_ascii=False)
    os.replace(staging, live_path)  # atomic on same filesystem
    return {"committed": True, "path": live_path,
            "stats": views.get("stats", {})}


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kb_compiler.records.common import load_json, save_json
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--invalidate-paper", default="")
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args()
    recs = load_json(args.records, {})
    views = load_json(args.views, {})
    prov = build_provenance_index(views)
    pids = [p for p in args.invalidate_paper.split(",") if p]
    filtered, report = invalidate(recs, paper_ids=pids)
    blast = {}
    for rid in report["record_ids"]:
        for loc in prov.get(rid, []):
            blast[loc["view"]] = blast.get(loc["view"], 0) + 1
    report["blast_radius_by_view"] = blast
    report["provenance_index_size"] = len(prov)
    save_json(report, args.out_report)
    print(json.dumps({k: report[k] for k in
                      ("n", "by_paper", "blast_radius_by_view", "provenance_index_size")},
                     ensure_ascii=False), flush=True)
