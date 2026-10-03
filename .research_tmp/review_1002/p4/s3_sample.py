# -*- coding: utf-8 -*-
"""P4 stage 3 (v2, strict): per-system candidate pool (arXiv-resolvable) -> regex pre-filter
(drop table rows/list bullets/multi-study aggregates) -> LLM strict filter: sentence attributes a
specific MEASURED empirical outcome to THIS single cited paper's own experiments. Sample up to
N per system, one sentence per paper, fixed seed. Output: s3_sample.json"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import P4, chat  # noqa: E402

N_PER = 14
pairs = json.load(open(os.path.join(P4, "s1_pairs.json"), encoding="utf-8"))
res = json.load(open(os.path.join(P4, "s2_resolved.json"), encoding="utf-8"))
t2id, meta = res["t2id"], res["id_meta"]
BAD = re.compile(r"\||^\s*[-*#]|\bstudies\b|\bstudy found\b.*\bstudies\b|\breviewed\b|\d+ out of \d+", re.I)


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


SYS = ("You screen sentences from AI-written research reports. Mark keep=true ONLY if ALL hold: "
       "(a) the sentence attributes a specific MEASURED empirical outcome (a performance comparison, "
       "an effect of an intervention, an observed behavior/trend from experiments or data analysis) "
       "to the ONE cited paper; (b) the outcome plausibly comes from that paper's OWN experiments "
       "(not a survey summarizing others, not a theorem/proof, not an algorithmic complexity bound); "
       "(c) it is not a definition, design description, motivation, or quote of an open question. "
       "Be strict: when unsure, keep=false.")
out = {}
rng = random.Random(42)
for sysname, v in pairs.items():
    pool, seen = [], set()
    for x in v:
        aid = x["arxiv_id"] or t2id.get(norm(x["title"]))
        s = x["sent"]
        if not aid or BAD.search(s) or not (60 <= len(s) <= 700):
            continue
        aid = re.sub(r"v\d+$", "", aid)
        if (s[:120], aid) in seen:
            continue
        seen.add((s[:120], aid))
        pool.append({**x, "arxiv_id": aid,
                     "paper_title": (meta.get(aid) or {}).get("title") or x["title"],
                     "paper_date": (meta.get(aid) or {}).get("date")})
    rng.shuffle(pool)
    picked, used, i = [], set(), 0
    while len(picked) < N_PER and i < len(pool):
        batch = [p for p in pool[i:i + 20] if p["arxiv_id"] not in used]
        i += 20
        if not batch:
            continue
        lst = "\n".join(f"{k}. [cites: {b['paper_title']}] {b['sent']}" for k, b in enumerate(batch))
        r = chat('Return JSON {"labels":[{"i":0,"keep":false},...]} for every sentence.\n\n' + lst,
                 system=SYS, max_tokens=3000)
        for lab in r.get("labels", []):
            k = lab.get("i")
            if isinstance(k, int) and 0 <= k < len(batch) and lab.get("keep"):
                b = batch[k]
                if b["arxiv_id"] in used or len(picked) >= N_PER:
                    continue
                used.add(b["arxiv_id"])
                picked.append(b)
    out[sysname] = {"pool": len(pool), "picked": picked}
    print(sysname, "pool", len(pool), "picked", len(picked), flush=True)
json.dump(out, open(os.path.join(P4, "s3_sample.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
