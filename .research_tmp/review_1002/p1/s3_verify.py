# -*- coding: utf-8 -*-
"""Step 3/4: sample claims, generate queries, Sciverse search, DeepSeek counterexample judging."""
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

OUT = C.P1
SEED = 20261003

QGEN = """A research report contains this literature-level claim:
CLAIM: "{claim}"
ORIGINAL SENTENCE: "{sentence}"
REPORT QUESTION: "{question}"

1. claim_date: the point in time at which the sentence asserts the gap holds. If the sentence frames it as
   current (present tense, "remains", "to our knowledge", no date), use "2025-05". If it is explicitly historical
   (e.g. "as of 2018", "prior to 2020", "at the time"), give that year as "YYYY".
2. counterexample: one sentence defining precisely what a published paper must do to directly refute the claim
   (the specific method/task/setting; not merely related work).
3. queries: 4 diverse academic search queries (keywords, synonyms, aliases, alternative terminology) that a
   careful researcher would issue to find such a paper.
4. framing: "current_assertion" if the report itself asserts the gap holds now; "attributed_historical" if the
   sentence reports a cited paper's own past motivation/novelty statement (e.g. "S was built because prior work
   was limited", "the authors note no prior work ...") or otherwise describes a past state.
5. described_work: name of the specific work/system the sentence itself describes (whose own claim this may be),
   or "" if none.
Return JSON {{"claim_date": "...", "counterexample": "...", "queries": ["...", "...", "...", "..."],
 "framing": "...", "described_work": "..."}}"""

JUDGE = """CLAIM from a research report: "{claim}"
ORIGINAL SENTENCE: "{sentence}"
The claim is asserted to hold as of: {claim_date}
A paper directly refutes the claim if: {counterexample}
EXCLUDE as counterexample the work the sentence itself describes or cites as the source of the statement
({described_work}) — it cannot refute its own motivation statement.

Candidate papers (title, year, abstract excerpt):
{cands}

For each candidate decide if it DIRECTLY refutes the claim: the abstract must show the paper actually does the
specific thing (not just a related topic, not a generic survey mention, not a future-work statement), AND it
must be published before the claim date (year strictly earlier, or same year if the claim date is 2025-05 and
the year is 2025 - mark those "same_year").
Return JSON {{"verdict": "refuted"|"not_refuted"|"unclear",
 "refuting": [{{"idx": <int>, "why": "<=25 words", "same_year": true|false}}],
 "note": "<=30 words"}}
"refuted" only if at least one candidate clearly qualifies. "unclear" if a candidate plausibly qualifies but the
abstract is insufficient to tell."""


def norm_t(t):
    return re.sub(r"[^a-z0-9 ]", " ", (t or "").lower().replace("title:", "")).split()


def dedup_claims(res):
    seen = {}
    for r in res:
        if r["label"] not in ("strong", "weak", "open_problem"):
            continue
        sysg = "ours" if r["sys"] in ("ours", "ours_supp") else r["sys"]
        key = (sysg, " ".join(norm_t(r["claim"]))[:160])
        if key not in seen:
            seen[key] = {**r, "sysg": sysg}
    return list(seen.values())


def sample(claims):
    rng = random.Random(SEED)
    out = []
    for sysg in sorted({c["sysg"] for c in claims}):
        for lab, cap in (("strong", 25), ("weak", 10), ("open_problem", 6)):
            pool = [c for c in claims if c["sysg"] == sysg and c["label"] == lab]
            rng.shuffle(pool)
            out += pool[:cap]
    return out


def verify(c):
    try:
        g = C.chat(QGEN.format(claim=c["claim"], sentence=c["sentence"][:600],
                               question=(c.get("question") or "n/a")[:300]), max_tokens=900,
                   thinking=False)
    except Exception as e:  # noqa: BLE001
        return {**c, "verdict": "error", "err": str(e)}
    cd = str(g.get("claim_date") or "2025-05")
    m = re.match(r"(\d{4})", cd)
    cyear = int(m.group(1)) if m else 2025
    queries = (g.get("queries") or [])[:4]
    cands, seen = [], set()
    for qi, q in enumerate(queries):
        rows = C.search(q, k=8)
        for rank, row in enumerate(rows):
            if row.get("error") or not row.get("title"):
                continue
            key = " ".join(norm_t(row["title"]))[:120]
            if key in seen:
                continue
            seen.add(key)
            y = row.get("year")
            try:
                y = int(y) if y is not None else None
            except Exception:
                y = None
            if y is not None and y > cyear:
                continue  # post-dates claim
            cands.append({**row, "year": y, "q_idx": qi, "rank": rank})
    known = [x for x in cands if x["year"] is not None][:20]
    unknown_year = [x for x in cands if x["year"] is None]
    if not known:
        return {**c, "claim_date": cd, "queries": queries, "counterexample": g.get("counterexample"), "framing": g.get("framing"), "described_work": g.get("described_work"),
                "verdict": "not_refuted", "n_cands": 0, "n_unknown_year": len(unknown_year),
                "refuting": [], "note": "no dated candidates"}
    block = "\n".join(f"[{i}] {x['title']} ({x['year']}) :: {(x.get('abstract') or '')[:700]}"
                      for i, x in enumerate(known))
    try:
        j = C.chat(JUDGE.format(claim=c["claim"], claim_date=cd, sentence=c["sentence"][:600],
                                described_work=g.get("described_work") or "none",
                                counterexample=g.get("counterexample"), cands=block), max_tokens=8000)
    except Exception as e:  # noqa: BLE001
        return {**c, "verdict": "error", "err": str(e)}
    ref = []
    for x in j.get("refuting") or []:
        try:
            k = int(x.get("idx"))
            p = known[k]
            ref.append({"title": p["title"], "year": p["year"], "abstract": p.get("abstract", "")[:900],
                        "q_idx": p["q_idx"], "rank": p["rank"], "why": x.get("why"),
                        "same_year": bool(x.get("same_year"))})
        except Exception:
            pass
    return {**c, "claim_date": cd, "queries": queries, "counterexample": g.get("counterexample"),
            "framing": g.get("framing"), "described_work": g.get("described_work"),
            "verdict": j.get("verdict"), "refuting": ref, "note": j.get("note"),
            "n_cands": len(known), "n_unknown_year": len(unknown_year)}


def main():
    res = json.load(open(os.path.join(OUT, "s2_classified.json"), encoding="utf-8"))
    claims = dedup_claims(res)
    from collections import Counter
    print("unique claims", Counter((c["sysg"], c["label"]) for c in claims))
    samp = sample(claims)
    print("sampled", len(samp), Counter((c["sysg"], c["label"]) for c in samp))
    json.dump(claims, open(os.path.join(OUT, "s3_unique_claims.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(verify, samp))
    json.dump(out, open(os.path.join(OUT, "s3_verified.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(Counter((o["sysg"], o["label"], o["verdict"]) for o in out))
    print(C.calls())


if __name__ == "__main__":
    main()
