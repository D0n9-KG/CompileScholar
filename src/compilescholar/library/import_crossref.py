# -*- coding: utf-8 -*-
"""DOIs -> the registry via Crossref works metadata (api.crossref.org/works/<doi>, polite pool through sources.http):
title, authors (given + family), container (venue), and the dates of INTEGRATED-SYSTEM-1005 §2.1 — published-online
(kind online), issued (kind issued), published-print folded into issued when earlier, and the `received` assertion
when the publisher deposits one (kind received: priority only, never visibility). Used for papers outside arXiv
(other fields; a benchmark's DOI-only members) and to measure Crossref's received-date coverage."""
from __future__ import annotations

import json
import re
import time

from ..core import ids
from ..core.asof import Date
from ..sources import http
from . import identity, store


_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}


def assertion_date(v: str) -> Date | None:
    """Publisher assertion dates: ISO ('2015-07-01') or Springer / Nature style ('1 July 2015', 'July 2015')."""
    d = Date.parse(v)
    if d is not None and re.match(r"^\s*\d{4}", v or ""):
        return d
    m = re.fullmatch(r"\s*(?:(\d{1,2})\s+)?([A-Za-z]+)\.?\s+(\d{4})\s*", v or "")
    mo = _MONTHS.get(m.group(2).lower()) if m else None
    if not mo:
        return None
    return Date.parse(f"{m.group(3)}-{mo:02d}" + (f"-{int(m.group(1)):02d}" if m.group(1) else ""))


_TAG = re.compile(r"<[^>]+>")


def strip_markup(s: str) -> str:
    """Crossref titles and abstracts carry JATS / MathML / HTML tags ('NdV<mml:math>...<mml:mn>4</mml:mn>...'): keep the
    text content ('NdVO4'), decode entities, collapse whitespace."""
    import html
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", s or ""))).strip()


def _date(parts) -> Date | None:
    p = ((parts or {}).get("date-parts") or [[None]])[0]
    if not p or p[0] is None:
        return None
    return Date.parse("-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(p)))


def record(doi: str) -> identity.Incoming | None:
    d = ids.normalize_doi(doi)
    if not d:
        return None
    r = http.get("crossref", f"https://api.crossref.org/works/{d}")
    if r.status == 404 or not r.ok:
        return None
    m = r.json().get("message") or {}
    title = strip_markup(" ".join(m.get("title") or []))
    authors = [{"name": " ".join(x for x in (a.get("given"), a.get("family")) if x), "surname": a.get("family") or ""}
               for a in m.get("author") or [] if a.get("family")]
    dates = [("online", 0, _date(m.get("published-online"))), ("issued", 0, _date(m.get("issued")))]
    pp = _date(m.get("published-print"))
    if pp and (dates[1][2] is None or pp.hi < dates[1][2].hi):
        dates[1] = ("issued", 0, pp)
    for a in m.get("assertion") or []:
        if a.get("name") == "received" and a.get("value"):
            dates.append(("received", 0, assertion_date(a["value"])))
    return identity.Incoming(source="crossref", ids=[("doi", d, "self")], title=title,
                             abstract=strip_markup(m.get("abstract") or ""),
                             venue=" ".join(m.get("container-title") or []),
                             dates=[x for x in dates if x[2] is not None], authors=authors)


def run(dois: list[str], log=print, path=None) -> dict:
    """Fetch first (network, no lock held), then write under the registry lock."""
    n = {"imported": 0, "not_found": 0}
    t0 = time.time()
    incs = []
    for i, doi in enumerate(dois, 1):
        inc = record(doi)
        if inc is None:
            n["not_found"] += 1
        else:
            incs.append(inc)
        if i % 100 == 0:
            log(f"[library] crossref fetched {i}/{len(dois)}")
    with store.lock(path):
        con = store.connect(path)
        w = identity.Writer(con, "crossref")
        for inc in incs:
            w.add(inc)
            n["imported"] += 1
        n.update(w.close())
        n["undated"] = identity.recompute_first_public(con)
        con.execute("INSERT INTO imports(source, inputs, counts, started_at, finished_at) VALUES (?,?,?,?,?)",
                    ("crossref", json.dumps({"dois": len(dois)}), json.dumps(n),
                     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), time.strftime("%Y-%m-%dT%H:%M:%S")))
        con.commit()
        con.close()
    log(f"[library] crossref: {n}")
    return n
