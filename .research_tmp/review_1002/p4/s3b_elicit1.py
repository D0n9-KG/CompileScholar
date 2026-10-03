# -*- coding: utf-8 -*-
"""P4 stage 3b: re-sample Elicit restricted to sentences citing exactly ONE paper (fixes the
multi-citation pairing artifact found in manual review). Same strict filter as s3.
Writes s3b_elicit1.json and fetches HTML."""
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from common import P4, calls, chat, fetch_html  # noqa: E402

N_PER = 14
pairs = json.load(open(os.path.join(P4, "s1_pairs.json"), encoding="utf-8"))["elicit"]
res = json.load(open(os.path.join(P4, "s2_resolved.json"), encoding="utf-8"))
t2id, meta = res["t2id"], res["id_meta"]
BAD = re.compile(r"\||^\s*[-*#]|\bstudies\b|\breviewed\b|\d+ out of \d+|\bseveral\b|\banother\b|\bothers\b", re.I)
old = {p["arxiv_id"] for p in json.load(open(os.path.join(P4, "s3_sample.json"), encoding="utf-8"))["elicit"]["picked"]}


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


by_sent = defaultdict(set)
for x in pairs:
    by_sent[x["sent"]].add(x["title"])
SYS = ("You screen sentences from AI-written research reports. Mark keep=true ONLY if ALL hold: "
       "(a) the sentence attributes a specific MEASURED empirical outcome (a performance comparison, "
       "an effect of an intervention, an observed behavior/trend from experiments or data analysis) "
       "to the ONE cited paper; (b) the outcome plausibly comes from that paper's OWN experiments "
       "(not a survey summarizing others, not a theorem/proof); (c) it is not a definition, design "
       "description, motivation, or open question. Be strict: when unsure, keep=false.")
pool, seen = [], set()
for x in pairs:
    s = x["sent"]
    if len(by_sent[s]) != 1 or BAD.search(s) or not (60 <= len(s) <= 700):
        continue
    aid = t2id.get(norm(x["title"]))
    if not aid or aid in old or (s[:120], aid) in seen:
        continue
    seen.add((s[:120], aid))
    pool.append({**x, "arxiv_id": aid, "paper_title": (meta.get(aid) or {}).get("title") or x["title"],
                 "paper_date": (meta.get(aid) or {}).get("date"), "system": "elicit"})
random.Random(7).shuffle(pool)
picked, used, i = [], set(), 0
while len(picked) < N_PER and i < len(pool):
    batch = [p for p in pool[i:i + 20] if p["arxiv_id"] not in used]
    i += 20
    lst = "\n".join(f"{k}. [cites: {b['paper_title']}] {b['sent']}" for k, b in enumerate(batch))
    r = chat('Return JSON {"labels":[{"i":0,"keep":false},...]} for every sentence.\n\n' + lst,
             system=SYS, max_tokens=3000)
    for lab in r.get("labels", []):
        k = lab.get("i")
        if isinstance(k, int) and 0 <= k < len(batch) and lab.get("keep"):
            b = batch[k]
            if b["arxiv_id"] not in used and len(picked) < N_PER:
                used.add(b["arxiv_id"])
                picked.append(b)
for p in picked:
    p["html_chars"] = len(fetch_html(p["arxiv_id"]))
print("pool", len(pool), "picked", len(picked), "html_ok", sum(1 for p in picked if p["html_chars"]), calls())
json.dump(picked, open(os.path.join(P4, "s3b_elicit1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
