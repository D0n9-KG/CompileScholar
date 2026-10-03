# -*- coding: utf-8 -*-
"""P4 stage 2: resolve cited titles -> arXiv id via local OAI snapshot (one streaming pass).
Also records submission year-month for each resolved id. Output: s2_resolved.json"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import P4  # noqa: E402

SNAP = "//192.168.199.138/Share400T/pub/LLM_Data/data/JournalPapers/arXiv_Dataset/arxiv-metadata-oai-snapshot.json"


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


pairs = json.load(open(os.path.join(P4, "s1_pairs.json"), encoding="utf-8"))
want = {norm(x["title"]) for v in pairs.values() for x in v if x["title"] and not x["arxiv_id"]}
want = {t for t in want if len(t) >= 20}
ids_wanted = {x["arxiv_id"] for v in pairs.values() for x in v if x["arxiv_id"]}
print("unique titles to resolve:", len(want), "ids to date:", len(ids_wanted), flush=True)
t2id, id_meta = {}, {}
with open(SNAP, encoding="utf-8", errors="replace") as f:
    for line in f:
        d = json.loads(line)
        nt = norm(d.get("title"))
        aid = d.get("id")
        hit = nt in want
        if hit or aid in ids_wanted:
            if hit and nt not in t2id:
                t2id[nt] = aid
            id_meta[aid] = {"title": re.sub(r"\s+", " ", d.get("title", "")),
                            "date": ((d.get("versions") or [{}])[0].get("created") or ""),
                            "cats": d.get("categories")}
print("resolved titles:", len(t2id), flush=True)
json.dump({"t2id": t2id, "id_meta": id_meta}, open(os.path.join(P4, "s2_resolved.json"), "w",
                                                    encoding="utf-8"), ensure_ascii=False)
