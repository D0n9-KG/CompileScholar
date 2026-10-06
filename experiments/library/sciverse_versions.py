# -*- coding: utf-8 -*-
"""Which arXiv version is the text Sciverse holds, and can its date be trusted as that version's date?

For every multi-version paper found in the Sciverse coverage sample (sciverse_coverage.py): GROBID-parse v1 and the
latest version from the NAS mirror, take the sentences only v1 has and the sentences only the latest version has
(documents.versions.delta), and count how many of each appear in the Sciverse text (word-bigram coverage >= 0.8).
The version whose own sentences Sciverse contains is the version it holds; compared with the version its date names.
Also, over all ScholarCatalyst members: share with later versions and the v1 -> latest gap.
Run from PowerShell (UNC). Output: runs/sciverse_coverage/versions.json"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics

from compilescholar.core import paths
from compilescholar.documents import parse as P
from compilescholar.documents import tei as TE
from compilescholar.documents import versions as VV
from compilescholar.sources import sciverse as S

OUT = paths.runs() / "sciverse_coverage"


def bigrams(s: str) -> set:
    w = re.findall(r"[a-z0-9]+", s.lower())
    return set(zip(w, w[1:]))


def found_in(sents: list[dict], text_bg: set) -> tuple[int, int]:
    hit = 0
    n = 0
    for s in sents:
        b = bigrams(s["text"])
        if len(b) < 6:
            continue
        n += 1
        hit += len(b & text_bg) / len(b) >= 0.8
    return hit, n


def doc(aid: str, v: int) -> dict | None:
    p = paths.resource("arxiv_pdf_mirror") / aid.split(".")[0] / f"{aid}v{v}.pdf"
    if not p.exists():
        return None
    data = p.read_bytes()
    return TE.parse(P.grobid(data, P._url("grobid")), P.cite_links(data), f"{aid}v{v}").to_dict()


def main() -> None:
    rows = [json.loads(l) for l in open(OUT / "sample.jsonl", encoding="utf-8")]
    res = []
    for r in rows:
        vs = dict((int(a), b) for a, b in r.get("versions") or [])
        if not r.get("found") or len(vs) < 2 or not re.fullmatch(r"arxiv:\d{4}\.\d{4,5}", r["paper_id"]):
            continue
        aid, latest = r["paper_id"][6:], max(vs)
        v1d, vld = doc(aid, 1), doc(aid, latest)
        if not v1d or not vld:
            continue
        text = S.request_json("GET", "/content", query={"doc_id": r["doc_id"], "offset": 0, "limit": 400000},
                              timeout_seconds=120).get("text") or ""
        tb = bigrams(text)
        only_latest = VV.delta(v1d, vld)["sentences"]
        only_v1 = VV.delta(vld, v1d)["sentences"]
        h1, n1 = found_in(only_v1, tb)
        hl, nl = found_in(only_latest, tb)
        held = ("v1" if h1 / max(1, n1) > 0.5 and hl / max(1, nl) < 0.5 else
                f"v{latest}" if hl / max(1, nl) > 0.5 and h1 / max(1, n1) < 0.5 else
                "both/neither")
        row = {"paper_id": r["paper_id"], "versions": vs, "sv_date": r.get("sv_date"),
               "date_says": r.get("sv_matches"), "v1_only_found": [h1, n1], "latest_only_found": [hl, nl],
               "held": held, "sv_chars": len(text)}
        res.append(row)
        print(json.dumps(row), flush=True)
    con = sqlite3.connect(f"file:{(paths.library() / 'registry.sqlite').as_posix()}?mode=ro", uri=True)
    gaps, multi, n = [], 0, 0
    for pid, in con.execute("SELECT paper_id FROM members WHERE benchmark='scholarcatalyst'"):
        vs = con.execute("SELECT version, hi FROM dates WHERE paper_id=? AND kind='arxiv_v' ORDER BY version",
                         (pid,)).fetchall()
        if not vs:
            continue
        n += 1
        if len(vs) > 1:
            multi += 1
            import datetime as dt
            gaps.append((dt.date.fromisoformat(vs[-1][1]) - dt.date.fromisoformat(vs[0][1])).days)
    q = statistics.quantiles(gaps, n=10)
    summary = {"papers": res, "sc_members_with_arxiv_dates": n, "multi_version": multi,
               "gap_days_median": statistics.median(gaps), "gap_days_deciles": [round(x) for x in q],
               "gap_over_365": sum(g > 365 for g in gaps)}
    (OUT / "versions.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "papers"}), flush=True)


if __name__ == "__main__":
    main()
