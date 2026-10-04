# -*- coding: utf-8 -*-
"""Rubric-track judge (official asta-bench sqa/rubric.py judge prompt VERBATIM),
scored against the auto-generated Multi rubrics (multi_rubric_gen.py).

Per (arm, qid): builds the official joint-assessment prompt (question +
response + enumerated criteria with examples), GLM-5.3 scores each criterion
0/1/2, weighted mean = rubric score. Gold probe included by default (user
directive: the auto-generated rubric must be validated by the gold answer's
ceiling — if gold doesn't score high, the rubric is invalid).

Scores append to answers/<arm>.rubric_scores.jsonl (resume-safe, last-row-wins).

Usage: python multi_rubric_judge.py [--gold-probe] [--interval 300]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "..", "src")
sys.path.insert(0, _SRC)

_MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "scholarqa_multi")
os.environ["LLM_CALL_LOG"] = os.path.join(_MULTI, "judge", "ledger_judge.jsonl")
os.environ.setdefault("LLM_RUN_ID", "multi-judge")
os.environ.setdefault("LLM_SOCK_TIMEOUT", "300")
os.environ.setdefault("LLM_WALL_TIMEOUT", "600")

from kb_infra.llm import call_paratera, parse_json_response  # noqa: E402
from multi_baseline_common import load_questions  # noqa: E402
import multi_rubric_gen as GEN  # noqa: E402

ARMS = {
    "ours": os.path.join(_MULTI, "baselines", "ours", "answers_ours.json"),
    "lightrag": os.path.join(_MULTI, "baselines", "lightrag", "answers_lightrag.json"),
    "paperqa": os.path.join(_MULTI, "baselines", "paperqa", "answers_paperqa.json"),
}
GOLD_OUT = os.path.join(_MULTI, "judge", "rubric_gold_scores.jsonl")
RUBRICS = GEN.OUT
MODEL = os.environ.get("RUBRIC_JUDGE_MODEL", "GLM-5.3")
WORKERS = int(os.environ.get("RUBRIC_JUDGE_WORKERS", "8"))

# ---- official judge prompts (astabench/evals/sqa/rubric.py, verbatim) ----
SYSTEM_PROMPT = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.
You will then be given an enumerated list of criteria by which to evaluate the response. Each criterion specifies requirements that the answer must satisfy. You will assign a score accordingly (see below).
You will also be given a list of examples (in <examples></examples> tags, below each criterion) that illustrate the type of details that would satisfy the criterion. We do NOT expect any of the specified details to necessarily appear in the answer. These are strictly to be used as guidance for locating the answers that satisfy the set requirement.

For each criterion, return a score of 0, 1 or 2 indicating how appropriate the response is based on the given criterion. 0 means the response does not meet the criterion, 1 means the response somewhat meets the criterion, 2 means the response perfectly meets the criterion. Judge only the specified aspect(s) delimited by the criterion, not any other qualities of the answer.

Return your result as a JSON object with a single key `scores` whose value is a list of objects, each having keys `criteria_idx`, `reasoning`, `score` and `evidence` from the text supporting the claim."""


def _criteria_block(ingredients: list) -> str:
    lines = []
    for n, ing in enumerate(ingredients, 1):
        ex = "\n".join(str(e) for e in (ing.get("examples") or [])[:4])
        lines.append(f"<criterion>\n{n}. {ing['criterion']}\n<examples>\n{ex}\n</examples>\n</criterion>")
    return "\n".join(lines)


def judge_answer(question: str, response: str, ingredients: list) -> dict | None:
    """One official joint-assessment call -> weighted rubric score."""
    user_prompt = (f"<question>{question}</question>\n"
                   f"<response>{response}</response>\nCriteria:\n"
                   f"{_criteria_block(ingredients)}")
    parsed = None
    for _ in range(2):
        raw = call_paratera(SYSTEM_PROMPT + "\n\n" + user_prompt,
                            model=MODEL, max_tokens=6000, temperature=0.0,
                            enable_thinking=False)
        obj = parse_json_response(raw or "")
        if isinstance(obj, dict) and obj.get("scores"):
            parsed = obj["scores"]
            break
    if not parsed:
        return None
    # official validation semantics (rubric.py _validate_joint_assessment_payload):
    # every criteria_idx 1..N exactly once
    idxs = sorted(s.get("criteria_idx") for s in parsed
                  if isinstance(s, dict) and isinstance(s.get("criteria_idx"), int))
    if idxs != list(range(1, len(ingredients) + 1)):
        return None
    w = {n: float(ing.get("weight") or 0)
         for n, ing in enumerate(ingredients, 1)}
    tot_w = sum(w.values()) or 1.0
    num = 0.0
    per = []
    for s in parsed:
        i = s["criteria_idx"]
        sc = s.get("score")
        sc = 2 if sc == "PERFECTLY_MET" else 1 if sc == "PARTIALLY_MET" else \
            0 if sc == "UNMET" else (int(sc) if isinstance(sc, (int, float)) else 0)
        num += sc * w[i]
        per.append({"i": i, "score": sc})
    return {"rubric_score": round(num / tot_w / 2, 4),  # 0..1 normalized
            "per_criterion": per}


def _arm_rows(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    out = {}
    for r in json.load(open(path, encoding="utf-8")):
        qid = r.get("qid")
        ans = (r.get("answer_official_all")
               or r.get("answer_official_first") or "")
        if qid and ans.strip() and not r.get("err"):
            out[qid] = ans
    return out


def score_arm(arm: str, path: str, rubrics: dict, gold_q: dict) -> list[str]:
    sp = path.rsplit(".", 1)[0] + ".rubric_scores.jsonl"
    scored = set()
    if os.path.exists(sp):
        for l in open(sp, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    scored.add(json.loads(l)["qid"])
                except Exception:
                    pass
    answers = _arm_rows(path)
    newly = []
    for qid, ans in answers.items():
        if qid in scored or qid not in rubrics:
            continue
        res = judge_answer(gold_q[qid]["input"], ans[:9000],
                           rubrics[qid]["ingredients"])
        if res:
            newly.append({"qid": qid, "arm": arm, **res})
    if newly:
        with open(sp, "a", encoding="utf-8") as f:
            for r in newly:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [r["qid"] for r in newly]


def gold_probe(rubrics: dict, gold_q: dict, force: bool = False) -> list[str]:
    """Score the GOLD answers under the generated rubrics — the rubric-validity
    ceiling measurement (auto-rubric acceptance gate)."""
    scored = set()
    if os.path.exists(GOLD_OUT) and not force:
        for l in open(GOLD_OUT, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    scored.add(json.loads(l)["qid"])
                except Exception:
                    pass
    newly = []
    todo = [(qid, r) for qid, r in rubrics.items() if qid not in scored]
    def one(item):
        qid, r = item
        res = judge_answer(r["question"], gold_q[qid]["output"][:9000],
                           r["ingredients"])
        return (qid, res) if res else None
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for got in ex.map(one, todo):
            if got:
                newly.append({"qid": got[0], "arm": "gold", **got[1]})
    if newly:
        with open(GOLD_OUT, "a", encoding="utf-8") as f:
            for r in newly:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [r["qid"] for r in newly]


def summarize() -> dict:
    out = {}
    for arm in ("gold", "ours", "lightrag", "paperqa"):
        sp = (GOLD_OUT if arm == "gold"
              else ARMS[arm].rsplit(".", 1)[0] + ".rubric_scores.jsonl")
        rows = {}
        if os.path.exists(sp):
            for l in open(sp, encoding="utf-8"):
                l = l.strip()
                if l:
                    try:
                        r = json.loads(l)
                        rows[r["qid"]] = r
                    except Exception:
                        pass
        if rows:
            s = [r["rubric_score"] for r in rows.values()]
            out[arm] = {"n": len(rows), "rubric_mean": round(sum(s) / len(s), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-probe", action="store_true", default=True)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gold_q = {q["id"]: q for q in load_questions(GEN.QFILE)}
    while True:
        rubrics = {}
        if os.path.exists(RUBRICS):
            for r in json.load(open(RUBRICS, encoding="utf-8")):
                rubrics[r["case_id"]] = r
        print(f"[rubric-judge] rubrics available: {len(rubrics)}/108", flush=True)
        # gold probe first (rubric-validity gate)
        new = gold_probe(rubrics, gold_q)
        if new:
            s = (summarize().get("gold") or {})
            print(f"[rubric-judge] GOLD +{len(new)} (n={s.get('n')} "
                  f"ceiling={s.get('rubric_mean')})", flush=True)
        for arm, path in ARMS.items():
            try:
                new = score_arm(arm, path, rubrics, gold_q)
            except json.JSONDecodeError:
                continue
            if new:
                s = (summarize().get(arm) or {})
                print(f"[rubric-judge] {arm}: +{len(new)} "
                      f"(n={s.get('n')} rubric={s.get('rubric_mean')})",
                      flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
