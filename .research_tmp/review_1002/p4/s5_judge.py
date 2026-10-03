# -*- coding: utf-8 -*-
"""P4 stage 5: per item, DeepSeek extracts the design matrix of the supporting experiment(s)
and judges the report sentence's scope vs that design. Output: s5_judged.json"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import HTML_DIR, P4, calls, chat  # noqa: E402

KW_EXP = re.compile(r"experiment|evaluat|result|setup|setting|implementation|benchmark|dataset|"
                    r"analysis|study|empirical|ablation|finding|method", re.I)
KW_LIM = re.compile(r"limitation|discussion|conclusion|threat|future work|broader", re.I)
STOP = set("the a an of in on for to and or with by is are was were that this from as at be it its "
           "their they which than more most can also".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]{3,}", s.lower()) if w not in STOP}


def excerpt(text, sent, cap=42000):
    secs = re.split(r"\n§§ ", text)
    head = secs[0][:1500]
    abstract = next((s for s in secs if s.lower().startswith("abstract")), "")[:3500]
    intro = next((s for s in secs if re.match(r"\d*\.?\s*introduction", s.lower())), "")[:3000]
    exp = [s for s in secs if KW_EXP.search(s.split("\n", 1)[0])]
    lim = [s for s in secs if KW_LIM.search(s.split("\n", 1)[0])]
    st = toks(sent)
    paras = [p for p in re.split(r"\n", text) if len(p) > 120]
    rel = sorted(paras, key=lambda p: -len(st & toks(p)))[:14]
    parts = ["[TITLE/HEAD]\n" + head, "[ABSTRACT]\n" + abstract, "[INTRO]\n" + intro,
             "[PASSAGES MOST RELATED TO THE REPORT SENTENCE]\n" + "\n".join(p[:1200] for p in rel)]
    budget = cap - sum(len(p) for p in parts) - 7000
    exp_txt = ""
    for s in exp:
        if len(exp_txt) > budget:
            break
        exp_txt += "\n§§ " + s[:9000]
    lim_txt = "".join("\n§§ " + s[:3000] for s in lim)[:6500]
    return "\n\n".join(parts + ["[EXPERIMENT/RESULT SECTIONS]" + exp_txt[:max(budget, 0)],
                                "[LIMITATION/DISCUSSION/CONCLUSION SECTIONS]" + lim_txt])


SYS = ("You are a careful meta-scientist auditing whether an AI-written research report states a "
       "cited paper's finding within the scope that the paper's experiments actually tested.")

PROMPT = """REPORT SENTENCE (written by an AI research assistant, citing the paper below):
<<<{sent}>>>

CITED PAPER: {title} (arXiv {aid})
PAPER EXCERPTS:
<<<{exc}>>>

Tasks:
1. Locate the experiment(s)/analysis in the paper that support the finding the sentence attributes to it. If the paper does not contain that finding at all, say so.
2. Build the DESIGN MATRIX of that experiment: factors that were VARIED (with tested values/ranges) and factors held FIXED (e.g., model family/size, language, dataset/domain, task type, training scale, population/sample, hardware, metric). For each fixed factor, say whether the paper explicitly acknowledges it as a limitation/scope restriction.
3. Judge the REPORT SENTENCE against the design matrix. Overreach means the sentence, as worded, asserts the finding for a broader population/condition than tested along a specific dimension WITHOUT hedging (e.g. tested only on English, sentence says "LLMs" generally do X; tested only on 7B models, sentence says the effect holds for language models). Naming the tested method/task in general terms without claiming broader coverage is NOT overreach. Hedged wording ("in one study", "on benchmark X") that matches the tested scope is in_scope.
   Labels: in_scope | overreach_fixed (broadened along a FIXED factor) | overreach_varied (extrapolated beyond the tested range of a VARIED factor) | misrepresented (the finding is distorted: wrong direction/magnitude/causal claim) | misattributed (the cited paper does not report this finding) | unverifiable (excerpts insufficient).
4. Did the paper's OWN abstract already state this finding more broadly than its design supports (author self-generalization)?
5. hidden_limitation: is there at least one FIXED factor that is critical to the sentence's claim and that the paper does NOT acknowledge as a limitation?

Return JSON:
{{"support_location": "...", "varied": [{{"factor": "...", "values": "..."}}], "fixed": [{{"factor": "...", "value": "...", "acknowledged_as_limitation": true}}], "label": "...", "overreach_dimension": "... or null", "tested_scope": "one sentence", "claimed_scope": "one sentence", "rationale": "2-3 sentences quoting the key evidence", "author_abstract_overreach": false, "author_overreach_note": "...", "hidden_limitation": false, "hidden_limitation_factor": "... or null"}}"""


def judge(item):
    text = open(os.path.join(HTML_DIR, item["arxiv_id"].replace("/", "_") + ".txt"), encoding="utf-8").read()
    if not text:
        return {**item, "judge": {"label": "no_text"}}
    exc = excerpt(text, item["sent"])
    try:
        j = chat(PROMPT.format(sent=item["sent"], title=item.get("paper_title") or item.get("title"),
                               aid=item["arxiv_id"], exc=exc), system=SYS, max_tokens=5000)
    except Exception as e:  # noqa: BLE001
        j = {"label": "error", "err": str(e)[:200]}
    return {**item, "judge": j, "excerpt_chars": len(exc)}


if __name__ == "__main__":
    d = json.load(open(os.path.join(P4, "s3_sample.json"), encoding="utf-8"))
    items = [{**p, "system": s} for s, v in d.items() for p in v["picked"]]
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(judge, items))
    json.dump(out, open(os.path.join(P4, "s5_judged.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print(Counter((o["system"], o["judge"].get("label")) for o in out))
    print(calls())
