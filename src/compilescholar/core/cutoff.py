# -*- coding: utf-8 -*-
"""Knowledge cutoff: the single implementation shared by every arm.

Official CS2: dev/test retrieval tools use inserted_before="2025-05" (astabench/evals/sqa/task.py:492-497), i.e. only
literature published before 2025-05-01. Our sources mostly give a year only, so:
  - full date (YYYY-MM[-DD]) -> strictly before the cutoff month;
  - year only -> year < cutoff year passes; year == cutoff year cannot be decided, so it is excluded (conservative);
    year > cutoff year is excluded;
  - no year -> excluded.
The cutoff comes from a per-thread value (DSB uses a different cutoff per question while answering concurrently) and
falls back to KNOWLEDGE_CUTOFF (YYYY-MM). Unset means no filtering.
"""
from __future__ import annotations

import os
import re
import threading

_DATE = re.compile(r"^(\d{4})(?:-(\d{1,2}))?")
_TL = threading.local()


def set_thread_cutoff(v: str | None):
    """Per-thread cutoff; None falls back to the environment variable."""
    _TL.value = v


def raw_cutoff() -> str:
    v = getattr(_TL, "value", None)
    return (v if v is not None else os.environ.get("KNOWLEDGE_CUTOFF") or "").strip()


def cutoff() -> tuple[int, int] | None:
    m = _DATE.match(raw_cutoff())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 1)


def allowed(year=None, date: str | None = None, cut: tuple[int, int] | None = None) -> bool:
    """True = usable. `cut` defaults to the current cutoff; no cutoff set means always True."""
    cut = cut if cut is not None else cutoff()
    if cut is None:
        return True
    cy, cm = cut
    if date:
        m = _DATE.match(str(date))
        if m:
            y, mo = int(m.group(1)), (int(m.group(2)) if m.group(2) else None)
            if mo is not None:
                return (y, mo) < (cy, cm)
            year = y
    try:
        y = int(year) if year is not None else None
    except (TypeError, ValueError):
        y = None
    if y is None:
        return False
    if y < cy:
        return True
    if y > cy:
        return False
    return cm > 12  # same year, month unknown: excluded


def filter_rows(rows: list[dict], year_key: str = "year", date_key: str | None = None) -> tuple[list[dict], int]:
    """Filter candidate rows; returns (kept rows, number excluded by the cutoff)."""
    if cutoff() is None:
        return rows, 0
    kept = [r for r in rows if allowed(r.get(year_key), r.get(date_key) if date_key else None)]
    return kept, len(rows) - len(kept)
