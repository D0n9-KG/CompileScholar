# -*- coding: utf-8 -*-
"""Part 2: survey-claimed absence records -> was the gap already addressed before the survey's year?"""
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from s3_verify import norm_t  # noqa: E402

B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb"
N = int(os.environ.get("P1_N", "30"))
NQ = int(os.environ.get("P1_NQ", "3"))

QGEN = """A survey paper published in {year} states this research gap (verbatim quote + normalized gap):
SUBJECT: "{subject}"
GAP: "{missing}"
QUOTE: "{quote}"

1. checkable: true if the gap is a concrete, checkable literature claim (e.g. "no dataset for X", "no method
   handles Y", "X has not been studied"); false if it is vague (generic "more research needed", "scalability is an
   issue", an inherent limitation of a method, or a performance shortfall rather than an absence of work).
2. counterexample: one sentence defining precisely what a paper must do to show this gap had ALREADY been
   addressed (directly; not merely related).
3. queries: {nq} diverse academic search queries to find such a paper.
Return JSON {{"checkable": true|false, "counterexample": "...", "queries": [...]}}"""

JUDGE = """A survey published in {year} claims this gap: "{missing}" (subject: {subject}).
The gap was already addressed if a paper: {counterexample}

Candidates published in or before {year} (title, year, abstract excerpt):
{cands}

Does any candidate DIRECTLY and clearly do this, published STRICTLY before {year} (or in {year} — mark same_year)?
Return JSON {{"verdict": "already_done"|"not_found"|"unclear",
 "refuting": [{{"idx": <int>, "why": "<=25 words", "same_year": true|false}}], "note": "<=30 words"}}"""


def survey_years():
    yrs = {}
    for f in ("survey_manifest.json", "manifest_all.json", "hub_manifest.json"):
        d = json.load(open(os.path.join(B, f), encoding="utf-8"))
        rows = d if isinstance(d, list) else list(d.values())
        for r in rows:
            if isinstance(r, dict) and r.get("paper_id") and r.get("year"):
                try:
                    yrs.setdefault(r["paper_id"], int(str(r["year"])[:4]))
                except Exception:
                    pass
    return yrs


def year_of(pid, yrs):
    if pid in yrs:
        return yrs[pid]
    m = re.match(r"arxiv_(\d{2})(\d{2})\.", pid)
    return 2000 + int(m.group(1)) if m else None


def run(item):
    pid, r, y = item
    try:
        g = C.chat(QGEN.format(year=y, subject=r.get("subject"), missing=r.get("missing"),
                               quote=(r.get("quote") or "")[:400], nq=NQ), max_tokens=900,
                   thinking=False)
    except Exception as e:  # noqa: BLE001
        return {"pid": pid, "verdict": "error", "err": str(e)}
    base = {"pid": pid, "year": y, "subject": r.get("subject"), "missing": r.get("missing"),
            "gap_type": r.get("gap_type"), "checkable": g.get("checkable"),
            "counterexample": g.get("counterexample"), "queries": (g.get("queries") or [])[:NQ]}
    if not g.get("checkable"):
        return {**base, "verdict": "not_checkable"}
    cands, seen = [], set()
    for qi, q in enumerate(base["queries"]):
        for rank, row in enumerate(C.search(q, k=8)):
            if row.get("error") or not row.get("title"):
                continue
            key = " ".join(norm_t(row["title"]))[:120]
            if key in seen:
                continue
            seen.add(key)
            try:
                yy = int(row.get("year")) if row.get("year") is not None else None
            except Exception:
                yy = None
            if yy is None or yy > y:
                continue
            cands.append({**row, "year": yy, "q_idx": qi, "rank": rank})
    cands = cands[:20]
    if not cands:
        return {**base, "verdict": "not_found", "n_cands": 0, "refuting": []}
    block = "\n".join(f"[{i}] {x['title']} ({x['year']}) :: {(x.get('abstract') or '')[:650]}"
                      for i, x in enumerate(cands))
    try:
        j = C.chat(JUDGE.format(year=y, missing=r.get("missing"), subject=r.get("subject"),
                                counterexample=g.get("counterexample"), cands=block), max_tokens=8000)
    except Exception as e:  # noqa: BLE001
        return {**base, "verdict": "error", "err": str(e)}
    ref = []
    for x in j.get("refuting") or []:
        try:
            p = cands[int(x.get("idx"))]
            ref.append({"title": p["title"], "year": p["year"], "abstract": p.get("abstract", "")[:900],
                        "q_idx": p["q_idx"], "rank": p["rank"], "why": x.get("why"),
                        "same_year": bool(x.get("same_year"))})
        except Exception:
            pass
    return {**base, "verdict": j.get("verdict"), "refuting": ref, "note": j.get("note"),
            "n_cands": len(cands)}


def main():
    R = json.load(open(os.path.join(B, "records_merged.json"), encoding="utf-8"))
    yrs = survey_years()
    pool = [(pid, r, year_of(pid, yrs)) for pid, p in R.items() for r in p.get("records", [])
            if r.get("kind") == "absence" and r.get("absence_type") == "survey_claimed"]
    pool = [x for x in pool if x[2]]
    rng = random.Random(20261003)
    rng.shuffle(pool)
    samp = pool[:N]
    print("pool", len(pool), "sample", len(samp))
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(run, samp))
    json.dump(out, open(os.path.join(C.P1, "s5_survey_gaps.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    from collections import Counter
    print(Counter(o["verdict"] for o in out))
    print(C.calls())


if __name__ == "__main__":
    main()
