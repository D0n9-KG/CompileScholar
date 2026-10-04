# -*- coding: utf-8 -*-
"""Prereg ruling ③: classify the 108 questions into the "unexpected load"
subset (forum-tone / non-polished queries) with TRANSPARENT rules, for user
review before the prereg freezes (keyword + human-read double confirmation).

Rule families (a question enters the subset if it matches ANY):
  C1 forum-tone: greetings, "has anyone", first-person singular, contractions,
     thanks, casual ellipsis
  C2 conversational ask / imperative paper-finding: "Can/Could you explain",
     "Find papers", "recommendations on", "share papers", "provide references"
  C3 existential literature probing: "Is/Are there any", "Has there been",
     "Is it possible", "Are there related works"
  C4 informal style (MANUAL list from full read-through): terse lowercase
     phrasing and/or typos ("orgnelle", "recognzie", "messager", ...) —
     not regex-derivable; each entry human-verified against the question text

Gold-blind discipline: classification uses ONLY q["input"] (question text),
never gold answers or ctxs.

Usage: python classify_unexpected.py  -> prints table, writes
       UNEXPECTED-SUBSET.md
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

C1 = [
    (r"\b[Hh]i everyone\b", "greeting"),
    (r"\b[Hh]as anyone\b|\b[Hh]ave anyone\b", "has-anyone"),
    (r"\b[Cc]an someone\b", "can-someone"),
    (r"\bwould appreciate\b", "would-appreciate"),
    (r"\bthanks!?\b", "thanks"),
    (r"\bI\b|\bI'm\b|\bmy\b|\bme\b", "first-person"),
    (r"\b(can't|don't|isn't|I've|won't|it's|doesn't|aren't)\b", "contraction"),
]
C2 = [
    (r"^[Cc](an|ould) you (explain|please)", "can-you-explain"),
    (r"^[Ff]ind (some )?papers?\b", "find-papers"),
    (r"\brecommendations on\b", "recommendations"),
    (r"\bshare papers?\b", "share-papers"),
    (r"\bprovide some references\b", "provide-references"),
]
C3 = [
    (r"\b[Ii]s there any\b|\b[Aa]re there any\b", "is-there-any"),
    (r"\b[Hh]as there been\b|\b[Hh]ave there been\b", "has-there-been"),
    (r"\b[Ii]s it possible\b", "is-it-possible"),
    (r"\b[Aa]re there related works\b", "related-works"),
]
# C4: manual, from the full 108-question read-through (09-21): terse/casual
# phrasing and/or misspellings that mark non-polished queries
C4 = {
    "pan_biophysics_3": "typos: orgnelle/hetergeniety; terse",
    "pan_biophysics_4": "typo: recognzie; sentence-fragment style",
    "pan_biophysics_6": "lowercase terse: 'how cancer cell response to anti-caner drugs'",
    "pan_biophysics_7": "typo-spacing: 'findings(novelty )'",
    "pan_biophysics_9": "typo: reprensentative; article-free phrasing",
    "minyang_physics_2": "typo: messager",
    "minyang_physics_4": "typo: multi-messager",
    "minyang_physics_6": "typo: optomechincal; fragment",
    "yanyu_photonics_6": "typo: all-dielectri; fragment",
    "yanyu_photonics_8": "typo: usinhg",
    "rulin_cs_3": "typo: envolve (+ 'Could you explain' = C2 too)",
    "weijia_cs_2": "typo: multilmodal; 'What are papers that...' reference-list ask",
    "akari_cs_5": "'viz-a-viz' casual + recommendation ask (C2 too)",
    "akari_cs_7": "lowercase opening 'do we know how true...'; multi-line casual",
    "rulin_cs_1": "'What are the latest works on...' reference-list ask",
}


def classify(q):
    text = q["input"]
    qid = q["id"]
    hits = []
    for pat, tag in C1:
        if re.search(pat, text):
            hits.append(f"C1:{tag}")
    for pat, tag in C2:
        if re.search(pat, text):
            hits.append(f"C2:{tag}")
    for pat, tag in C3:
        if re.search(pat, text):
            hits.append(f"C3:{tag}")
    if qid in C4:
        hits.append("C4:manual")
    return hits


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    items = json.load(open(os.path.join(BASE, "data", "scholarqa_multi.json"),
                           encoding="utf-8"))
    rows = []
    for q in items:
        hits = classify(q)
        rows.append((q["id"], q.get("subject"), bool(hits), "; ".join(hits),
                     q["input"]))
    inn = [r for r in rows if r[2]]
    lines = []
    lines.append(f"# 未预期负载子集分类清单（预注册裁点③，09-21）\n")
    lines.append(f"总数：**{len(inn)}/108**（规则 C1 论坛口吻 / C2 会话式索取 / "
                 f"C3 存在性探询 / C4 人工判定的非正式文体）\n")
    lines.append(f"gold-blind：仅用题面 input 分类，未接触 gold 答案与 ctxs。\n")
    lines.append("| # | id | subject | 规则命中 | 题面（截断 110 字符） |")
    lines.append("|---|----|---------|----------|----------------------|")
    for i, (qid, subj, _, hits, text) in enumerate(inn, 1):
        t = text.replace("\n", " ").replace("|", "\\|")[:110]
        lines.append(f"| {i} | {qid} | {subj} | {hits} | {t} |")
    by_subj = {}
    for _, subj, hit, _, _ in rows:
        by_subj.setdefault(subj, [0, 0])
        by_subj[subj][1] += 1
        if hit:
            by_subj[subj][0] += 1
    lines.append("\n按学科分布：")
    for subj in sorted(by_subj):
        n, d = by_subj[subj]
        lines.append(f"- {subj}: {n}/{d}")
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(BASE, "UNEXPECTED-SUBSET.md"), "w",
              encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
