# -*- coding: utf-8 -*-
"""Step 2: DeepSeek normalizes + classifies regex candidates into atomic negative-existence claims."""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
from s1_candidates import NEG, sents, sec_text, B, OUT  # noqa: E402

SYS = ("You analyze sentences from AI-generated research reports. You decide whether a sentence "
       "asserts something about the EXISTENCE or ABSENCE of published research (a literature-level "
       "negative-existence claim). Be strict and literal.")

PROMPT = """For each numbered sentence (with the report's research question for context), output a JSON object
{{"items": [{{"i": <number>, "label": "strong"|"weak"|"open_problem"|"not_literature", "claim": "...", "attributed": true|false}}]}}

Labels:
- "strong": asserts that NO (or essentially no) published work/study/method exists that does X, or that X has not been explored/studied/addressed/evaluated, or "to our/the best of knowledge, none ...".
- "weak": asserts research on X is few/limited/scarce/underexplored/lacking (a quantity claim, not a strict non-existence claim).
- "open_problem": asserts a specific technical question/problem remains open/unsolved (no known solution/result).
- "not_literature": anything else (model/system capability statements, physical facts, generic 'challenges', user behaviour, process not well understood by users, a past gap that the sentence says WAS later solved, etc.).

"claim": for strong/weak/open_problem, rewrite as ONE atomic, self-contained, searchable claim of the form
"As of early 2025, no published work <does X>" (strong), "As of early 2025, few published works <do X>" (weak), or
"As of early 2025, no published result <resolves problem P>" (open_problem). X/P must be specific (name the method/task/setting), resolving pronouns from context. Empty string for not_literature.
"attributed": true if the sentence attributes the gap statement to a cited source (e.g. "[3] notes that...", "the authors state", or a citation marker right after the gap phrase), else false.

Research question context and sentences:
{block}
"""


def build_pool():
    d = json.load(open(os.path.join(OUT, "s1_candidates.json"), encoding="utf-8"))
    pool = []
    for s, cs in d.items():
        for c in cs:
            pool.append(c)
    # supplementary ours pool (other batches, dedup vs main ours)
    main = {c["sentence"] for c in d.get("ours", [])}
    seen = set(main)
    for f in sorted(glob.glob(os.path.join(B, "judge_input_ours_batch*.json"))):
        if f.endswith(("34e.json", "32b.json")) or ".v1." in f:
            continue
        for qi, r in enumerate(json.load(open(f, encoding="utf-8"))):
            for sn in sents(sec_text(r.get("sections") or [])):
                if NEG.search(sn) and sn[:700] not in seen:
                    seen.add(sn[:700])
                    pool.append({"sys": "ours_supp", "qi": r["qid"], "question": r["question"],
                                 "sentence": sn[:700], "batch": os.path.basename(f)})
    for i, c in enumerate(pool):
        c["cid"] = i
    return pool


def run_batch(items):
    block = "\n".join(f"{k+1}. [Q: {(c.get('question') or 'n/a')[:300]}]\n   SENTENCE: {c['sentence']}"
                      for k, c in enumerate(items))
    try:
        r = C.chat(PROMPT.format(block=block), system=SYS, max_tokens=3000)
        got = {int(x["i"]): x for x in r.get("items", []) if str(x.get("i", "")).isdigit() or isinstance(x.get("i"), int)}
    except Exception as e:  # noqa: BLE001
        got = {}
        print("batch fail", e)
    out = []
    for k, c in enumerate(items):
        x = got.get(k + 1, {})
        out.append({**c, "label": x.get("label", "error"), "claim": x.get("claim", ""),
                    "attributed": x.get("attributed")})
    return out


def main():
    pool = build_pool()
    print("pool", len(pool))
    batches = [pool[i:i + 8] for i in range(0, len(pool), 8)]
    res = []
    with ThreadPoolExecutor(4) as ex:
        for out in ex.map(run_batch, batches):
            res += out
    json.dump(res, open(os.path.join(OUT, "s2_classified.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    from collections import Counter
    c = Counter((r["sys"], r["label"]) for r in res)
    for k in sorted(c):
        print(k, c[k])
    print(C.calls())


if __name__ == "__main__":
    main()
