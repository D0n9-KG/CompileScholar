# -*- coding: utf-8 -*-
"""GOLDCOV step 4: judge whether each gold point appears in OUR terminal
answer (ours53). Combined with step 3 this yields the 2x2 decomposition
(KB coverage x answer coverage).

Usage:
  py -3.13 judge_answer.py            # resume-safe, writes ans_cov.jsonl
"""
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = os.path.dirname(HERE)
SRC = "C:/Users/D0n9/Desktop/CompileScholar/src"
sys.path.insert(0, SRC)
from kb_infra.llm import call_paratera  # noqa: E402

ANSWERS = os.environ.get("GOLDCOV_ANSWERS") or os.path.join(
    D2, "term_stage", "term_53arm", "answers_pilot_term_ours53.json")
OUT = os.environ.get("GOLDCOV_ANS_OUT") or os.path.join(HERE, "ans_cov.jsonl")
MODEL = "GLM-5.3"
WORKERS = 8

PROMPT = """You are auditing whether ONE atomic key point from a reference answer also appears in a model's answer to the same question.

Key point: {point}

Model's answer:
{answer}

Decide:
- "yes": the model's answer contains the substance of this point (numbers must match or be compatible; a range containing the exact value does NOT count unless the value itself is stated).
- "no": the point's substance is absent from the model's answer.

Return ONLY JSON: {{"in_answer": true/false, "reason": "<one sentence>"}}"""


def parse_verdict(raw):
    t = (raw or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        try:
            import json5
            obj = json5.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict) or not isinstance(obj.get("in_answer"), bool):
        return None
    return {"in_answer": obj["in_answer"],
            "reason": str(obj.get("reason", ""))[:300]}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    answers = json.load(open(ANSWERS, encoding="utf-8"))
    amap = ({a["id"]: a.get("answer", "") for a in answers}
            if isinstance(answers, list) else dict(answers))
    points = json.load(open(os.path.join(HERE, "points.json"), encoding="utf-8"))
    done = {}
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["key"]] = r
    todo = []
    for qid, q in points.items():
        if qid not in amap:
            print(f"[WARN] no answer for {qid}", flush=True)
            continue
        for p in q["points"]:
            key = f"{qid}#{p['point_id']}"
            if key not in done and "__DECOMPOSE_FAIL__" not in p["text"]:
                todo.append((qid, q, p, key))
    print(f"{len(done)} judged, {len(todo)} to go (workers={WORKERS})", flush=True)
    lock = threading.Lock()
    out_f = open(OUT, "a", encoding="utf-8")
    n_done = [0]

    def work(item):
        qid, q, p, key = item
        ans = amap.get(qid, "")
        if isinstance(ans, dict):
            ans = ans.get("answer", "")
        prompt = PROMPT.format(point=p["text"], answer=str(ans)[:14000])
        v = None
        for _ in range(3):
            raw = call_paratera(prompt, MODEL, max_tokens=500,
                                enable_thinking=False, temperature=0.0)
            v = parse_verdict(raw)
            if v:
                break
        if not v:
            v = {"in_answer": None, "reason": "judge_parse_fail"}
        rec = {"key": key, "qid": qid, "prompt_type": q["prompt_type"],
               "point_id": p["point_id"], "ptype": p["ptype"], "point": p["text"], **v}
        with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done[0] += 1
            if n_done[0] % 25 == 0:
                print(f"[{n_done[0]}/{len(todo)}]", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for _ in as_completed(futs):
            pass
    y = sum(1 for l in open(OUT, encoding="utf-8") if l.strip()
            and json.loads(l).get("in_answer"))
    n = sum(1 for l in open(OUT, encoding="utf-8") if l.strip())
    print(f"DONE: yes={y}/{n} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
