# -*- coding: utf-8 -*-
"""E1b labels: two models of different families judge each sampled (entry, resolved paper) pair blind, plus 40
negative controls (entry paired with a different sampled paper) to check annotators do not just say Y.
Annotator 1 = DeepSeek-V4.1-Flash, annotator 2 = Kimi-K2.6 (both via Paratera). Writes
runs/e1b-resolution-20261005/{labels.tsv,agreement.json}."""
import concurrent.futures as cf
import json
import random
import re
import sys
from pathlib import Path

from compilescholar.llm.client import call_paratera
from compilescholar.llm.jsonparse import parse_json_response

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "e1b-resolution-20261005"
MODELS = ("DeepSeek-V4.1-Flash", "Kimi-K2.6")
PROMPT = """You are checking a citation-resolution system. A paper's bibliography entry was resolved to an arXiv paper.
Decide whether the resolved paper is the SAME work the bibliography entry describes (a preprint and its published
version count as the same work).

Bibliography entry:
{entry}

Resolved arXiv paper: {resolved} — {title}

Answer with JSON only: {{"label": "Y" | "N" | "?", "reason": "<one short sentence>"}}
Y = same work, N = different work, ? = the information is insufficient to tell."""


def label(model, x):
    for _ in range(3):
        raw = call_paratera(PROMPT.format(entry=x["entry"][:800], resolved=x["resolved"], title=x["resolved_title"]),
                            model=model, max_tokens=300, enable_thinking=False)
        o = parse_json_response(raw or "")
        if isinstance(o, dict) and o.get("label") in ("Y", "N", "?"):
            return o["label"]
    return "ERR"


def main():
    rows = [json.loads(l) for l in open(OUT / "sample.jsonl", encoding="utf-8")]
    rng = random.Random(1005)
    neg = []
    for x in rng.sample(rows, 40):
        other = rng.choice([y for y in rows if y["resolved"] != x["resolved"]])
        neg.append({**x, "id": x["id"] + "-NEG", "resolved": other["resolved"], "resolved_title": other["resolved_title"]})
    items = rows + neg
    with cf.ThreadPoolExecutor(12) as ex:
        labs = {m: list(ex.map(lambda x, m=m: label(m, x), items)) for m in MODELS}
    with open(OUT / "labels.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tstyle\tmethod\t" + "\t".join(MODELS) + "\n")
        for i, x in enumerate(items):
            f.write(f"{x['id']}\t{x['style']}\t{x['method']}\t" + "\t".join(labs[m][i] for m in MODELS) + "\n")
    n = len(rows)
    pos = {m: labs[m][:n] for m in MODELS}
    negl = {m: labs[m][n:] for m in MODELS}
    both_y = sum(1 for a, b in zip(*pos.values()) if a == "Y" and b == "Y")
    agree = sum(1 for a, b in zip(*pos.values()) if a == b)
    disagree = [(rows[i]["id"], pos[MODELS[0]][i], pos[MODELS[1]][i]) for i in range(n) if pos[MODELS[0]][i] != pos[MODELS[1]][i]]
    out = {"n": n, "both_Y": both_y, "precision_both_Y": round(both_y / n, 3), "agreement": agree,
           "per_model_Y": {m: sum(v == "Y" for v in pos[m]) for m in MODELS},
           "by_style": {s: {m: sum(1 for i in range(n) if rows[i]["style"] == s and pos[m][i] == "Y") for m in MODELS}
                        | {"n": sum(1 for x in rows if x["style"] == s)} for s in ("numeric", "author-year")},
           "by_method": {k: {m: sum(1 for i in range(n) if rows[i]["method"] == k and pos[m][i] == "Y") for m in MODELS}
                         | {"n": sum(1 for x in rows if x["method"] == k)} for k in sorted({x["method"] for x in rows})},
           "negative_controls": {m: {"n": len(neg), "said_Y": sum(v == "Y" for v in negl[m])} for m in MODELS},
           "disagreements": disagree}
    json.dump(out, open(OUT / "agreement.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
