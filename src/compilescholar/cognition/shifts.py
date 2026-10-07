# -*- coding: utf-8 -*-
"""Reception-shift events at T (phase D reader): reads the materialised shift_event table — the deterministic
screen (effect floors + Fisher exact + BH-FDR 0.05, cognition.build.screen_shifts) followed by the LLM
SHIFT_VERIFY adjudication. Only status='approved' rows are events; a row becomes visible when its evidence
window has closed by T (window_end <= T — an open window would show a shift the field has not finished
making). The old midpoint-split rule reader is retired (measured ~75% false positives).

Event facets: became_component / became_baseline / recategorized / limitation_exposed / superseded. The
`direction` field carries the facet's shape (early/late shares + the Fisher p, or from/to categories, or the
replaces date for superseded); `evidence` carries the verbatim quotes the screen captured."""
from __future__ import annotations

import json

from .asof import AsOf

FACETS = ("became_component", "became_baseline", "recategorized", "limitation_exposed", "superseded")


def _j(x, default):
    try:
        v = json.loads(x) if x else default
    except (TypeError, ValueError):
        return default
    return v if isinstance(v, type(default)) else default


def _row(r) -> dict:
    return {"subject": r[0], "type": r[1], "window": [r[2], r[3]], "date": r[3],
            "direction": _j(r[4], {}), "evidence": _j(r[5], []), "decided_by": r[6]}


def events(view: AsOf, pid: str | None = None, facet: str | None = None) -> list[dict]:
    """Approved shift events visible at T, date-ordered. pid/facet narrow (None = all subjects)."""
    if view.cog is None:
        return []
    q = ("SELECT subject, facet, window_start, window_end, direction, evidence, decided_by FROM shift_event "
         "WHERE status='approved' AND window_end<=?")
    a: list = [view.T]
    if pid is not None:
        q += " AND subject=?"
        a.append(pid)
    if facet is not None:
        q += " AND facet=?"
        a.append(facet)
    return [_row(r) for r in view.cog.execute(q + " ORDER BY window_end, subject, facet", a)]
