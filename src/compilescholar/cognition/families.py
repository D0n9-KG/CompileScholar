# -*- coding: utf-8 -*-
"""Method families at T (phase D reader): reads the materialised Leiden time-sliced snapshots
(cognition.build — cumulative co-citation + lineage + shared-category edge weights, fixed seed, weighted
RBConfigurationVertexPartition = modularity's built-in hub de-weighting; stubs are boundary nodes, never
members).

A query at T reads the newest snapshot whose date is <= T. The grid is monthly over the last 3 years of the
data, quarterly before that, capped at 15 years, plus explicit benchmark cut-offs; a T before the earliest
snapshot gets NO families and the answer says so — it is never clamped up to the earliest snapshot, which
would leak future grouping structure into the past.

Names: the LLM names the families of the latest snapshot (and benchmark cut-offs) and prunes weak boundary
members; older snapshots inherit a name by member overlap (Jaccard >= 0.3, named_by='inherit'). Membership
in an older snapshot is exactly what the field's evidence supported by that date — a member's visibility at
T >= snapshot date follows from the build's own visibility filter (first_hi <= snapshot)."""
from __future__ import annotations

import json

from .asof import AsOf


def snapshot_at(view: AsOf) -> tuple[str | None, str | None]:
    """(the snapshot to read at T, the earliest snapshot that exists). snapshot is None when T precedes the
    grid or the stage has no snapshots yet."""
    if view.cog is None:
        return None, None
    lo = view.cog.execute("SELECT min(snapshot) FROM family_snapshot").fetchone()[0]
    s = view.cog.execute("SELECT max(snapshot) FROM family_snapshot WHERE snapshot<=?", (view.T,)).fetchone()[0]
    return s, lo


def families(view: AsOf, scope: set | None = None, min_size: int = 2, k: int | None = None) -> dict:
    """Families of the snapshot at T. `scope` restricts members (a topic's seed set); families whose visible
    members drop below min_size are omitted. Families come in the build's size-descending order."""
    snap, lo = snapshot_at(view)
    out = []
    if snap:
        for fid, name, members, named_by in view.cog.execute(
                "SELECT family_id, name, members, named_by FROM family_snapshot WHERE snapshot=? "
                "ORDER BY family_id", (snap,)):
            mm = json.loads(members)
            if scope is not None:
                mm = [m for m in mm if m in scope]
            if len(mm) < min_size:
                continue
            out.append({"family_id": fid, "name": name, "named_by": named_by,
                        "members": mm, "n_members": len(mm)})
    if k:
        out = out[:k]
    return {"as_of": view.T, "snapshot": snap, "earliest_snapshot": lo,
            "before_grid": bool(lo and snap is None and view.T < lo), "families": out}


def family_of(view: AsOf, pid: str) -> dict | None:
    """The family a paper belongs to at T (None when it is in no family of that snapshot)."""
    snap, _ = snapshot_at(view)
    if not snap:
        return None
    for fid, name, members, named_by in view.cog.execute(
            "SELECT family_id, name, members, named_by FROM family_snapshot WHERE snapshot=? ORDER BY family_id",
            (snap,)):
        mm = json.loads(members)
        if pid in mm:
            return {"family_id": fid, "name": name, "named_by": named_by, "members": mm, "snapshot": snap}
    return None
