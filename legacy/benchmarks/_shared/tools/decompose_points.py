# -*- coding: utf-8 -*-
"""GOLDCOV step 2: decompose the 30 gold answers into atomic key points.

Each point must be independently verifiable against a knowledge-base record
(a fact, a number, a method property, a comparison, a trend claim, a stated
limitation...). Verbatim numbers stay verbatim.

Model: GLM-5.3 via Paratera, thinking disabled, temperature 0 (same source
as the terminal judge — one model, one provider for the whole instrument).

Usage:
  py -3.13 decompose_points.py            # all 30 questions
  py -3.13 decompose_points.py --qid PS-res-exp-xxxx   # single (resume aid)
"""
import argparse
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = os.path.dirname(HERE)
SRC = "C:/Users/D0n9/Desktop/CompileScholar/src"
sys.path.insert(0, SRC)
from kb_infra.llm import call_paratera  # noqa: E402

QUESTIONS = os.path.join(D2, "term_stage", "term_53arm", "pilot_questions30.json")
OUT = os.path.join(HERE, "points.json")
MODEL = "GLM-5.3"

PROMPT = """You are decomposing a reference answer (written by a domain expert from a set of research papers) into ATOMIC KEY POINTS for a coverage audit.

Rules:
- Each point must be a SINGLE verifiable claim (a fact, a number, a method property, a comparison between methods/numbers, a trend, a limitation, a dataset detail...).
- Keep all numbers VERBATIM (e.g. "65.70 ± 0.52", "+35.79 points", "4-bit").
- Keep method/dataset names verbatim (e.g. "NEFTune", "AlpacaEval", "LLaMA-2-7B").
- Points must be self-contained: understandable without seeing the question or other points.
- Split compound statements (setup + result, multiple numbers) into separate points.
- Do NOT add points that restate the question, summarize style, or give meta-commentary.
- Aim for completeness: every substantive statement in the reference answer becomes part of exactly one point. Minor connective prose can be dropped.
- Point types: fact | number | comparison | trend | limitation | dataset | method | other

Question (for context):
{question}

Reference answer:
{gold}

Return ONLY a JSON array:
[{{"point_id": 1, "text": "...", "ptype": "fact"}}, ...]"""


def parse_points(raw):
    t = (raw or "").strip()
    m = re.search(r"\[.*\]", t, re.S)
    if not m:
        return None
    body = m.group(0)
    try:
        arr = json.loads(body)
    except Exception:
        try:
            import json5
            arr = json5.loads(body)
        except Exception:
            arr = None
    if not isinstance(arr, list):
        return None
    out = []
    for i, p in enumerate(arr):
        if isinstance(p, dict) and isinstance(p.get("text"), str) and len(p["text"]) > 8:
            out.append({"point_id": len(out) + 1, "text": p["text"][:600],
                        "ptype": str(p.get("ptype", "other"))[:20]})
    return out or None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qs = json.load(open(QUESTIONS, encoding="utf-8"))["questions"]
    done = {}
    if os.path.exists(OUT):
        done = json.load(open(OUT, encoding="utf-8"))
    todo = [q for q in qs if q["qid"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    for i, q in enumerate(todo):
        prompt = PROMPT.format(question=q["question"][:3000], gold=q["gold_answer"][:14000])
        pts = None
        for attempt in range(3):
            raw = call_paratera(prompt, MODEL, max_tokens=6000,
                                enable_thinking=False, temperature=0.0)
            pts = parse_points(raw)
            if pts:
                break
        if not pts:
            print(f"[FAIL] {q['qid']} raw={str(raw)[:200]}", flush=True)
            pts = [{"point_id": 1, "text": "__DECOMPOSE_FAIL__", "ptype": "error"}]
        done[q["qid"]] = {"qid": q["qid"], "prompt_type": q["prompt_type"],
                          "question": q["question"], "gold_answer": q["gold_answer"],
                          "points": pts}
        json.dump(done, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        n = sum(len(v["points"]) for v in done.values())
        print(f"[{len(done)}/{len(qs)}] {q['qid']} ({q['prompt_type']}) "
              f"-> {len(pts)} points (total {n})", flush=True)
    n = sum(len(v["points"]) for v in done.values())
    print(f"DONE: {len(done)} questions, {n} points -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid")
    a = ap.parse_args()
    main()
