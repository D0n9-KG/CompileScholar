# -*- coding: utf-8 -*-
"""Time: dates with a precision, and the one visibility rule (INTEGRATED-SYSTEM-1005 v2 §2.2).

Every date is an interval: day precision lo = hi; month precision first..last day of the month; year precision
Jan 1..Dec 31. A thing dated D is visible at T (inclusive, a calendar day) iff D.hi <= T; when D.lo > T it is not
visible; in between (the precision cannot decide) it is NOT visible — the conservative rule the old cutoff module
applied to same-year, unknown-month records, now stated once for every precision. An undated thing is never visible
under a cutoff. Comparing raw strings is wrong ('2015' <= '2015-01-01' is True, a year-only date would leak), so
every filter compares `hi` as a full ISO day.

  Date.parse("2021")            -> 2021-01-01..2021-12-31 (year)
  Date.parse("2021-03")         -> 2021-03-01..2021-03-31 (month)
  Date.parse("2021-03-04T10:00") -> 2021-03-04 (day)
  AsOf("2021-06-30").visible(Date.parse("2021"))  -> False (undecidable)
  AsOf.before_month("2025-05")  -> T = 2025-04-30 (the CS2 convention: published before 2025-05-01)

External services only filter coarsely (Sciverse by year); `prefilter_year()` gives the widest year a service may be
asked for, and the caller re-checks every hit with `visible()` once its exact date is known."""
from __future__ import annotations

import calendar
import datetime as _dt
import re
from dataclasses import dataclass

_RX = re.compile(r"^\s*(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?")
PRECISIONS = ("day", "month", "year")


@dataclass(frozen=True)
class Date:
    lo: _dt.date
    hi: _dt.date
    precision: str

    @classmethod
    def parse(cls, v) -> "Date | None":
        """ISO-ish string (YYYY, YYYY-MM, YYYY-MM-DD, with any time suffix), int year, date/datetime; None if unusable."""
        if v is None or v == "":
            return None
        if isinstance(v, _dt.datetime):
            v = v.date()
        if isinstance(v, _dt.date):
            return cls(v, v, "day")
        if isinstance(v, int):
            return cls.year(v)
        m = _RX.match(str(v))
        if not m:
            return None
        y = int(m.group(1))
        if not 1000 <= y <= 2999:
            return None
        if m.group(2) is None:
            return cls.year(y)
        mo = int(m.group(2))
        if not 1 <= mo <= 12:
            return cls.year(y)
        if m.group(3) is None:
            return cls.month(y, mo)
        d = int(m.group(3))
        try:
            day = _dt.date(y, mo, d)
        except ValueError:
            return cls.month(y, mo)
        return cls(day, day, "day")

    @classmethod
    def year(cls, y: int) -> "Date":
        return cls(_dt.date(y, 1, 1), _dt.date(y, 12, 31), "year")

    @classmethod
    def month(cls, y: int, m: int) -> "Date":
        return cls(_dt.date(y, m, 1), _dt.date(y, m, calendar.monthrange(y, m)[1]), "month")

    @property
    def hi_iso(self) -> str:
        return self.hi.isoformat()

    @property
    def lo_iso(self) -> str:
        return self.lo.isoformat()

    def __str__(self) -> str:
        return {"day": self.lo.isoformat(), "month": self.lo.isoformat()[:7], "year": str(self.lo.year)}[self.precision]


def earliest(*dates: "Date | None") -> "Date | None":
    """The earliest of several dates for one thing (e.g. first public date over versions and sources): the one with the
    smallest hi; on a tie the more precise one."""
    ds = [d for d in dates if d is not None]
    if not ds:
        return None
    return min(ds, key=lambda d: (d.hi, PRECISIONS.index(d.precision)))


class AsOf:
    """A cutoff T (inclusive). AsOf(None) is "no cutoff": everything dated or not is visible."""

    def __init__(self, T: "str | _dt.date | None"):
        if T is None:
            self.T = None
        else:
            if isinstance(T, _dt.date):
                self.T = T
            else:
                try:
                    self.T = _dt.date.fromisoformat(str(T).strip())
                except ValueError:
                    raise ValueError(f"as_of must be YYYY-MM-DD, got {T!r}") from None

    @classmethod
    def before_month(cls, ym: str | None) -> "AsOf":
        """'YYYY-MM' meaning 'published before the first day of that month' (CS2 inserted_before) -> T = day before."""
        if not ym:
            return cls(None)
        m = re.match(r"^(\d{4})-(\d{1,2})", str(ym))
        if not m:
            raise ValueError(f"expected YYYY-MM, got {ym!r}")
        return cls(_dt.date(int(m.group(1)), int(m.group(2)), 1) - _dt.timedelta(days=1))

    @property
    def iso(self) -> str | None:
        return self.T.isoformat() if self.T else None

    def visible(self, d) -> bool:
        if self.T is None:
            return True
        if not isinstance(d, Date):
            d = Date.parse(d)
        return d is not None and d.hi <= self.T

    def state(self, d) -> str:
        """'visible' | 'invisible' | 'undecided' (precision too coarse) | 'undated' — for counting what was excluded."""
        if self.T is None:
            return "visible"
        if not isinstance(d, Date):
            d = Date.parse(d)
        if d is None:
            return "undated"
        if d.hi <= self.T:
            return "visible"
        return "invisible" if d.lo > self.T else "undecided"

    def prefilter_year(self) -> int | None:
        """Largest publication year an external service may return (its hits are re-checked with visible())."""
        return self.T.year if self.T else None

    def __repr__(self) -> str:
        return f"AsOf({self.iso!r})"


def legacy_month_rule(cut: str | None):
    """The frozen v9b rule (core.cutoff.allowed with an exclusive YYYY-MM cutoff), kept for the v9b arm so its
    characterization stays byte-identical: full date -> strictly before the cutoff month; year only -> earlier year
    passes, same year excluded. Returns allowed(year, date) -> bool."""
    from . import cutoff as _c
    m = re.match(r"^(\d{4})(?:-(\d{1,2}))?", cut or "")
    c = (int(m.group(1)), int(m.group(2) or 1)) if m else None
    return lambda year=None, date=None: _c.allowed(year, date, cut=c) if c else True
