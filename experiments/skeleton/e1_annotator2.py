# -*- coding: utf-8 -*-
"""E1 second annotator: a model from a different family (Kimi-K2.6 via Paratera) labels all 200 E1 items blind
(it sees the entry, the resolved title and the citing passage — never annotator 1's labels). Then agreement with
annotator 1 is computed. Writes results/e1/e1_labels_annotator2.tsv and e1_agreement.json."""
import concurrent.futures as cf
import json
import re
import sys

from compilescholar.core import paths
from compilescholar.llm.client import call_paratera
from compilescholar.llm.jsonparse import parse_json_response

OUT = paths.REPO / "results" / "e1"
MODEL = "Kimi-K2.6"
PROMPT = """You are checking a citation-resolution system. A survey paper cites a bibliography entry; the system resolved
that entry to a paper. Decide whether the resolved paper is the SAME paper the bibliography entry describes (same work;
a preprint and its published version count as the same paper).

Bibliography entry:
{entry}

Resolved paper: {resolved}

Answer with JSON only: {{"label": "Y" | "N" | "?", "reason": "<one short sentence>"}}
Y = same paper, N = different paper, ? = the information is insufficient to tell."""


def label(x):
    for _ in range(3):
        raw = call_paratera(PROMPT.format(entry=x["entry"], resolved=f"{x['paper']} — {x['resolved_title']}"),
                            model=MODEL, max_tokens=300, enable_thinking=False)
        o = parse_json_response(raw or "")
        if isinstance(o, dict) and o.get("label") in ("Y", "N", "?"):
            return x["id"], o["label"], re.sub(r"\s+", " ", str(o.get("reason") or ""))[:200]
    return x["id"], "ERR", ""


def main():
    rows = [json.loads(l) for l in open(OUT / "e1_sample.jsonl", encoding="utf-8")]
    with cf.ThreadPoolExecutor(8) as ex:
        res = sorted(ex.map(label, rows))
    with open(OUT / "e1_labels_annotator2.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tlabel\tnote\n")
        for i, l, r in res:
            f.write(f"{i}\t{l}\t{r}\n")
    a1 = {l.split("\t")[0]: l.split("\t")[1] for l in open(OUT / "e1_labels_annotator1.tsv", encoding="utf-8").read().splitlines()[1:]}
    a2 = {i: l for i, l, _ in res}
    blind = set(re.findall(r"^## (E1-\d{3})", (OUT / "e1_blind_review.md").read_text(encoding="utf-8"), re.M))
    def agree(ids):
        ids = [i for i in ids if a2[i] != "ERR"]
        return {"n": len(ids), "agree": sum(a1[i] == a2[i] for i in ids),
                "disagree": [(i, a1[i], a2[i]) for i in ids if a1[i] != a2[i]]}
    out = {"annotator2_model": MODEL, "all_200": agree(list(a2)), "blind_40": agree(sorted(blind)),
           "annotator2_counts": {k: sum(v == k for v in a2.values()) for k in ("Y", "N", "?", "ERR")}}
    json.dump(out, open(OUT / "e1_agreement.json", "w"), indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
