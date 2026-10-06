# -*- coding: utf-8 -*-
"""Probe a model's knowledge cutoff: does it know well-known arXiv papers first posted in a given month?

Per month from 2023-07 to 2026-06: the most-cited ScholarCatalyst-library arXiv papers of that month (cs, by citing
sentences in the old citations store), the model is asked for the first author and the main contribution given only
the exact title; a second call with a fabricated title of the same shape measures how often it claims to know papers
that do not exist. "Knows" = names the right first-author surname. The month where recall falls to the fabricated
rate is the effective cutoff (memorised papers, not general world knowledge).
Usage: python knowledge_cutoff_probe.py [provider] [model]"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import sys
from collections import defaultdict

from compilescholar.core import ids, paths
from compilescholar.llm import client as LC

PROMPT = """Do you know the paper titled "{title}"? If you know it, give its first author's surname and its main
contribution in one sentence. If you do not know this exact paper, say so; do not guess.
Answer as JSON: {{"know": true|false, "first_author_surname": "...", "contribution": "..."}}"""


def months():
    out = []
    for y in range(2023, 2027):
        for m in range(1, 13):
            if (y, m) >= (2023, 7) and (y, m) <= (2026, 6):
                out.append(f"{y}-{m:02d}")
    return out


def sample(per_month: int = 8) -> dict:
    reg = sqlite3.connect(f"file:{(paths.library() / 'registry.sqlite').as_posix()}?mode=ro", uri=True)
    cit = sqlite3.connect(f"file:{(paths.derived() / 'citations.sqlite').as_posix()}?mode=ro", uri=True)
    counts = defaultdict(int)
    for (cd,) in cit.execute("SELECT cited FROM cites WHERE cited LIKE 'paper:%'"):
        counts[cd[6:]] += 1
    by_month = defaultdict(list)
    for pid, fh in reg.execute("SELECT paper_id, first_hi FROM papers WHERE paper_id LIKE 'arxiv:2%' AND status='active' "
                               "AND first_kind='arxiv_v' AND first_hi >= '2023-07-01'"):
        by_month[fh[:7]].append((counts.get(pid[6:], 0), pid))
    out = {}
    for mo in months():
        top = [p for c, p in sorted(by_month.get(mo, []), reverse=True)[:per_month * 3] if c >= 1][:per_month]
        if len(top) < per_month:                        # recent months are barely cited yet: fill at random
            rest = [p for _, p in by_month.get(mo, []) if p not in top]
            top += random.Random(mo).sample(rest, min(per_month - len(top), len(rest)))
        rows = []
        for pid in top:
            t = reg.execute("SELECT title FROM records WHERE paper_id=? AND source='arxiv'", (pid,)).fetchone()
            a = reg.execute("SELECT names FROM authors WHERE paper_id=? AND source='arxiv'", (pid,)).fetchone()
            if t and a:
                rows.append({"paper_id": pid, "title": t[0], "surname": json.loads(a[0])[0]["surname"],
                             "cites": counts.get(pid[6:], 0)})
        out[mo] = rows
    return out


def fake(title: str, rng: random.Random) -> str:
    w = title.split()
    rng.shuffle(w)
    return " ".join(w)


def main(provider: str = "local", model: str | None = None) -> None:
    LC.configure({}, run_id="cutoff-probe", caller="probe")
    data = sample()
    rng = random.Random(1)
    res = {}
    for mo, rows in data.items():
        hit = fake_yes = 0
        for r in rows:
            o = LC.call_json(PROMPT.format(title=r["title"]), provider=provider, model=model, max_tokens=200) or {}
            if o.get("know") and ids.norm_title(o.get("first_author_surname") or "") == ids.norm_title(r["surname"]):
                hit += 1
            f = LC.call_json(PROMPT.format(title=fake(r["title"], rng)), provider=provider, model=model,
                             max_tokens=200) or {}
            fake_yes += bool(f.get("know"))
        res[mo] = {"n": len(rows), "knows_first_author": hit, "claims_to_know_fake": fake_yes,
                   "median_cites": sorted(r["cites"] for r in rows)[len(rows) // 2] if rows else 0}
        print(mo, json.dumps(res[mo]), flush=True)
    out = paths.runs() / "cutoff_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{provider}_{model or 'default'}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
