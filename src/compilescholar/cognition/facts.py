# -*- coding: utf-8 -*-
"""Family-level facts at T (phase D reader): reads the materialised fact groups (cognition.build —
deterministic recall by same-facet content-word Jaccard >= 0.3 inside one family, the LLM FACT_REL relation
verdict on candidate pairs, union-find over 'same', and the status timeline single-source -> established
(n_independent >= 2) -> consensus (n_independent >= 3 and >= 2 subjects), with a 'contested' event at the
date an 'opposite' verdict's statements become visible).

A fact at T is its member statements dated <= T (the extract store applies the cutoff); its status is the
last timeline event dated <= T, and `contested_since` is the first contradiction dated <= T (None when no
contradiction is visible yet). Independence counts come from the timeline events' evidence (author_key
greedy merge, computed at build time) — the reader never recomputes them."""
from __future__ import annotations

import json

from .asof import AsOf

FACT_FACETS = ("limitation", "method", "result", "categorization", "contribution")
_IN_CHUNK = 900          # sqlite variable limit headroom for the IN clauses


def _chunked(lst, n=_IN_CHUNK):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def facts(view: AsOf, members: list | None = None, family_id: str | None = None, facets=FACT_FACETS,
          k: int | None = None) -> list[dict]:
    """The facts stated about `members` (or one family's members), visible at T, strongest first."""
    if view.cog is None:
        return []
    if family_id is not None and members is None:
        row = view.cog.execute("SELECT members FROM family_snapshot WHERE family_id=? ORDER BY snapshot DESC "
                               "LIMIT 1", (family_id,)).fetchone()
        members = json.loads(row[0]) if row else []
    if members is None:
        return []
    facets = tuple(facets)
    sids: list[int] = []
    for part in _chunked(sorted(members)):
        ph = ",".join("?" * len(part))
        sids += [r[0] for r in view.ext.execute(
            f"SELECT id FROM statements WHERE about IN ({ph}) AND date<=? "
            f"AND facet IN ({','.join('?' * len(facets))})", (*part, view.T, *facets))]
    if not sids:
        return []
    fids: list[str] = []
    for part in _chunked(sids):
        ph = ",".join("?" * len(part))
        fids += [r[0] for r in view.cog.execute(
            f"SELECT DISTINCT fact_id FROM fact_member WHERE statement_id IN ({ph})", tuple(part))]
    if not fids:
        return []
    fids = sorted(set(fids))
    member_rows: dict[str, list] = {}
    for part in _chunked(fids):
        ph = ",".join("?" * len(part))
        for fid, sid, role in view.cog.execute(
                f"SELECT fact_id, statement_id, role FROM fact_member WHERE fact_id IN ({ph})", tuple(part)):
            member_rows.setdefault(fid, []).append((sid, role))
    status: dict[str, str] = {}
    counts: dict[str, dict] = {}
    contested: dict[str, str] = {}
    for part in _chunked(fids):
        ph = ",".join("?" * len(part))
        for fid, date, st, ev in view.cog.execute(
                f"SELECT fact_id, date, status, evidence FROM fact_status_event WHERE fact_id IN ({ph}) "
                "AND date<=? ORDER BY date, rowid", (*part, view.T)):
            if st == "contested":
                contested.setdefault(fid, date)
                continue
            status[fid] = st
            try:
                j = json.loads(ev) if ev else {}
            except (TypeError, ValueError):
                j = {}
            if isinstance(j, dict) and "n_independent" in j:
                counts[fid] = j
    all_ids = sorted({sid for rows in member_rows.values() for sid, _ in rows})
    stmts = {s["id"]: s for s in view.statements_by_id(all_ids)}       # date <= T applied by the view
    out = []
    for fid in fids:
        rows = member_rows.get(fid) or []
        sup = sorted((stmts[sid] for sid, _ in rows if sid in stmts), key=lambda s: (s["date"], s["id"]))
        if not sup or sup[0]["facet"] not in facets:
            continue                                  # nothing visible at T (or the facet was filtered out)
        rep = next((stmts[sid] for sid, role in rows if role == "representative" and sid in stmts), sup[0])
        ev = counts.get(fid) or {}
        about = sorted({s["about"] for s in sup})
        out.append({"fact_id": fid, "facet": rep["facet"], "text": rep["text"], "quote": rep["quote"],
                    "status": status.get(fid, "single-source"),
                    "contested_since": contested.get(fid),
                    "n_independent": ev.get("n_independent"), "n_subjects": len(about),
                    "members": about, "first_seen": sup[0]["date"],
                    "support": [{"by": s["speaker"], "about": s["about"], "date": s["date"],
                                 "kind": s["kind"], "epistemic": s["epistemic"], "quote": s["quote"]}
                                for s in sup[:8]]})
    out.sort(key=lambda f: (-(f["n_independent"] or 0), f["first_seen"], f["fact_id"]))
    return out[:k] if k else out
