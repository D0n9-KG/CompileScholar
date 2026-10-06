# -*- coding: utf-8 -*-
"""Merge-queue verdicts by LLM (INTEGRATED-SYSTEM-1005 §3: rules propose, an LLM decides; phase B gate: identity
merges spot-checked with two models).

Each open pair is shown to two models with both records' evidence: titles, abstracts (first 600 chars), authors,
dates with their sources, venues, identifiers. Question: the same work — one paper, possibly as preprint and published
version, with a revised title or abstract — or different works (companion papers Part I / Part II, a follow-up by the
same authors, a record that carries the other's DOI by mistake)? Both verdicts are stored in the queue row. A pair
merges only when both models say "same" with confidence >= MIN_CONF; both "different" -> rejected; anything else stays
open for review. The prompt never sees a benchmark, a question or a label."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from ..dfc.store import parallel
from ..llm import client as LC
from . import identity, store

MIN_CONF = 0.8
CHUNK = 2000
MODELS = ("local", "paratera")          # two different model families (Qwen local, DeepSeek on Paratera)
TEMPLATE = "library.same_work.v3"

PROMPT = """You decide whether two bibliographic records describe the same scholarly work.

Same work: one paper, possibly appearing as a preprint and as the published version (conference or journal), possibly
with a revised title, a lightly revised abstract, a different author order or a different date. A paper posted twice
to arXiv under two ids is one work.
Different works: companion papers (Part I / Part II); a later study by the same authors that re-runs the analysis on
new or extended data, or reports different numbers; a journal paper that substantially extends an earlier paper and
says so (new method parts, a new title); a survey and the paper it surveys; two papers with a generic or similar
title; a record that merely carries the other's identifier by mistake. Shared authors and overlapping wording alone
do not make two records the same work.

Record A
{a}

Record B
{b}

Answer with JSON only: {{"same": true|false, "confidence": 0.0-1.0, "reason": "<one sentence>"}}"""


def describe(con: sqlite3.Connection, pid: str) -> str:
    lines, abstract = [f"id: {pid}"], None
    # every source's title and venue; one abstract and one author list (arXiv's when present)
    for src, title, abs_, venue in con.execute(
            "SELECT source, title, abstract, venue FROM records WHERE paper_id=? ORDER BY source != 'arxiv', source",
            (pid,)):
        lines.append(f"[{src}] title: {title}")
        if venue:
            lines.append(f"[{src}] venue: {venue}")
        if abs_ and abstract is None:
            abstract = (src, abs_)
    if abstract:
        lines.append(f"[{abstract[0]}] abstract: {abstract[1][:600]}")
    for src, names in con.execute("SELECT source, names FROM authors WHERE paper_id=? ORDER BY source != 'arxiv' "
                                  "LIMIT 1", (pid,)):
        names = [a.get("name") or a.get("surname") for a in json.loads(names)]
        lines.append(f"[{src}] authors: {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")
    for src, kind, ver, hi, prec in con.execute(
            "SELECT source, kind, version, hi, precision FROM dates WHERE paper_id=? ORDER BY hi", (pid,)):
        lines.append(f"[{src}] date ({kind}{' v' + str(ver) if ver else ''}): {hi if prec == 'day' else hi[:4]}")
    idents = sorted(f"{s}:{v}" for s, v in con.execute("SELECT scheme, value FROM identifiers WHERE paper_id=?", (pid,))
                    if s in ("doi", "arxiv", "openalex", "s2"))
    if idents:
        lines.append("identifiers: " + ", ".join(idents))
    return "\n".join(lines)


def _valid(o) -> bool:
    return isinstance(o, dict) and isinstance(o.get("same"), bool) and isinstance(o.get("confidence"), (int, float))


def decide(verdicts: dict, two_arxiv_ids: bool = False) -> str:
    """merged | rejected | open, from {provider: verdict or None}. Two records with two different arXiv ids are two
    submissions; they are never merged automatically (the 200-pair audit's wrong and borderline merges were all of this
    kind: a later study by the same authors posted under a new id) — they stay separate, open for review."""
    vs = [verdicts.get(m) for m in MODELS]
    if any(v is None for v in vs):
        return "open"
    if all(not v["same"] for v in vs):
        return "rejected"
    if not two_arxiv_ids and all(v["same"] and v["confidence"] >= MIN_CONF for v in vs):
        return "merged"
    return "open"


def _two_arxiv_ids(con: sqlite3.Connection, a: str, b: str) -> bool:
    ax = [{v for (v,) in con.execute("SELECT value FROM identifiers WHERE scheme='arxiv' AND paper_id=?",
                                     (identity.canonical(con, p),))} for p in (a, b)]
    return bool(ax[0] and ax[1] and not ax[0] & ax[1])


def run(kinds=("shared_doi", "identifier_conflict", "title_match"), limit: int | None = None, workers: int = 32,
        apply: bool = True, only_ids: list[int] | None = None, log=print, path=None) -> dict:
    """Adjudicate open queue rows of `kinds` (or exactly the rows `only_ids`); verdicts already stored are reused.
    apply=False stores verdicts only (an audit run); apply=True also rejects and merges."""
    with store.lock(path):
        con = store.connect(path)
        rows = con.execute("SELECT id, a, b, kind, verdict FROM merge_queue WHERE status='open' AND kind IN (%s) "
                           "ORDER BY id" % ",".join("?" * len(kinds)), kinds).fetchall()
        if only_ids is not None:
            keep = set(only_ids)
            rows = [r for r in rows if r[0] in keep]
        rows = rows[:limit]
        log(f"[adjudicate] {len(rows):,} open pairs")
        n = {"merged": 0, "rejected": 0, "open": 0}
        merges = []
        # verdicts are written chunk by chunk, so an interrupted run resumes from the stored ones; merges are applied
        # at the end, after every verdict is in
        for c0 in range(0, len(rows), CHUNK):
            chunk = rows[c0:c0 + CHUNK]
            descs = {p: describe(con, identity.canonical(con, p)) for _, a, b, _, _ in chunk for p in (a, b)}
            lock, results = threading.Lock(), {}

            def one(row):
                qid, a, b, _, verdict = row
                got = json.loads(verdict) if verdict else {}
                if got.get("template") != TEMPLATE:          # verdicts of another prompt version are not reused
                    got = {"template": TEMPLATE}
                for prov in MODELS:
                    if got.get(prov) is None:
                        got[prov] = LC.call_json(PROMPT.format(a=descs[a], b=descs[b]), provider=prov,
                                                 max_tokens=300, validate=_valid, template=TEMPLATE, item=str(qid))
                with lock:
                    results[qid] = got
            parallel(one, chunk, workers)
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            for qid, a, b, kind, _ in chunk:
                got = results[qid]
                d = decide(got, _two_arxiv_ids(con, a, b))
                n[d] += 1
                con.execute("UPDATE merge_queue SET verdict=? WHERE id=?", (json.dumps(got, ensure_ascii=False), qid))
                if apply and d == "rejected":
                    con.execute("UPDATE merge_queue SET status='rejected', decided_by=?, decided_at=? WHERE id=?",
                                ("+".join(MODELS), now, qid))
                elif apply and d == "merged":
                    merges.append((a, b, kind))
            con.commit()
            log(f"[adjudicate] {min(c0 + CHUNK, len(rows)):,}/{len(rows):,} {n}")
        for a, b, kind in merges:
            identity.merge(con, a, b, reason=f"{kind}: two-model verdict", decided_by="+".join(MODELS),
                           recompute=False)
        if merges:
            identity.recompute_first_public(con)
        con.close()
    log(f"[adjudicate] {n}")
    return n
