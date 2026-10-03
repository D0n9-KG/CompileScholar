# -*- coding: utf-8 -*-
"""P4 stage 1: (report sentence, cited paper title / arXiv id) pairs per system.

Output: s1_pairs.json  {system: [{qid, sent, title, arxiv_id|None}]}
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import P4  # noqa: E402

CS2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
AX = re.compile(r"(?:arxiv\.org|ar5iv\.(?:labs\.arxiv\.)?org)/(?:abs|pdf|html)/(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})", re.I)


def sents(text):
    text = re.sub(r"\s+", " ", text or "")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(<*])", text) if len(s.strip()) > 40]


def openai_dr():
    out = []
    for x in json.load(open(os.path.join(CS2, "arm_memorized/sqa_openai_dr_dev.json"), encoding="utf-8")):
        for s in sents(x["answer"]):
            for u in re.findall(r"\]\((https?://[^)\s]+)\)", s):
                m = AX.search(u)
                if m:
                    clean = re.sub(r"\s*\(\[[^\]]*\]\([^)]*\)\)", "", s)
                    clean = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", clean)
                    out.append({"qid": x["question"][:60], "sent": clean, "title": None,
                                "arxiv_id": m.group(1)})
    return out


def _sections_pairs(secs, qid):
    out = []
    for s in secs:
        cmap = {c.get("id"): c for c in (s.get("citations") or [])}
        for st in sents(s.get("text", "")):
            for cid, c in cmap.items():
                if cid and cid in st and c.get("title"):
                    md = c.get("metadata") or {}
                    ax = md.get("arxiv") if isinstance(md, dict) else None
                    out.append({"qid": qid, "sent": st, "title": c["title"],
                                "arxiv_id": str(ax) if ax else None})
    return out


def harness():
    out = []
    d = json.load(open(os.path.join(CS2, "arm_harness/answers_harness_dev20.json"), encoding="utf-8"))
    for r in (d if isinstance(d, list) else d.values()):
        m = re.search(r"\{\s*\"sections\".*\}", r.get("result") or "", re.S)
        if not m:
            continue
        try:
            j = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            continue
        out += _sections_pairs(j.get("sections", []), r.get("qid", "")[:24])
    return out


def gptr():
    out = []
    d = json.load(open(os.path.join(CS2, "arm_gptr/answers_gptr_cs2.json"), encoding="utf-8"))
    for r in (d if isinstance(d, list) else d.values()):
        for s in r.get("sections", []):
            for st in sents(s.get("text", "")):
                for t, _u in re.findall(r"\[([^\]]{8,200})\]\((https?://[^)\s]+)\)", st):
                    clean = re.sub(r"\s*\(\[[^\]]*\]\([^)]*\)\)", "", st)
                    out.append({"qid": r.get("qid", "")[:24], "sent": clean, "title": t, "arxiv_id": None})
    return out


def elicit():
    out = []
    d = json.load(open(os.path.join(CS2, "arm_memorized/sqa_elicit_responses.json"), encoding="utf-8"))
    for r in (d if isinstance(d, list) else d.values()):
        for s in r.get("sections", []):
            cmap = {c.get("id"): ((c.get("metadata") or {}).get("paper") or {}).get("title")
                    for c in (s.get("citations") or [])}
            txt = s.get("text", "")
            for st in sents(txt):
                ids = re.findall(r'paperTitle="([^"]+)"', st)
                clean = re.sub(r"<Paper[^>]*></Paper>", "", st).strip()
                for pid in ids:
                    if cmap.get(pid):
                        out.append({"qid": r.get("id", "")[:40], "sent": clean, "title": cmap[pid],
                                    "arxiv_id": None})
    return out


if __name__ == "__main__":
    res = {"openai_dr": openai_dr(), "harness": harness(), "gptr": gptr(), "elicit": elicit()}
    for k, v in res.items():
        print(k, len(v), "with_arxiv:", sum(1 for x in v if x["arxiv_id"]))
    json.dump(res, open(os.path.join(P4, "s1_pairs.json"), "w", encoding="utf-8"), ensure_ascii=False)
