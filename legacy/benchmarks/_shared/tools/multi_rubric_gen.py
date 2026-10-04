# -*- coding: utf-8 -*-
"""Multi-108 rubric (ingredients) generation — user-approved 2026-09-23.

Multi has NO official rubric file (asta-bench ships rubrics_v1/v2 for the SQA
track only). Precedent for auto-generation: AI2's own follow-up work ("The
rubric generation is fully automated. Similar to recent TREC ... candidate
ingredients are pooled using claude-opus and are merged subsequently.") and
Lacuna (third party, published with auto rubrics via the official judge).

Pipeline per question (mirrors the official pooled-then-merged design):
  1. POOL: GLM-5.3 drafts candidate ingredients from the question + the
     question's own ctx passages (the official gold-answer source material —
     generation sees NO system answers, gold answers only as reference).
     3 independent drafts (temperature variation) for pooling diversity.
  2. MERGE: one GLM-5.3 call dedupes/merges into the final ingredient set,
     following the OFFICIAL schema (name/criterion/weight/examples), weights
     normalized, critical vs valuable tiers (official files carry both).
  3. Judge: the EXISTING official rubric.py judge prompt, verbatim (ported
     into judge_rubric.py), scores each answer 0-2 per ingredient.

All generation artifacts saved (judge/rubrics_generated.json + per-draft
forensics). Disclosed in the report as auto-generated rubrics.

Usage: python multi_rubric_gen.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "..", "src")
sys.path.insert(0, _SRC)

_MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "scholarqa_multi")
os.environ["LLM_CALL_LOG"] = os.path.join(_MULTI, "judge", "ledger_judge.jsonl")
os.environ.setdefault("LLM_RUN_ID", "multi-judge")
# GLM-5.3 rubric generation is a long-thinking call (6k max_tokens): the
# default 60s sock timeout killed every attempt at exactly 60s (measured:
# three ok=False rows, latency 60050ms each). Judge-loop verdicts are short
# and unaffected; generation needs the longer window.
os.environ.setdefault("LLM_SOCK_TIMEOUT", "300")
os.environ.setdefault("LLM_WALL_TIMEOUT", "600")

from kb_infra.llm import call_paratera, parse_json_response  # noqa: E402
from multi_baseline_common import load_questions  # noqa: E402

QFILE = os.path.join(_MULTI, "data", "scholarqa_multi.json")
OUT = os.path.join(_MULTI, "judge", "rubrics_generated.json")
DRAFTS_DIR = os.path.join(_MULTI, "judge", "rubric_drafts")
MODEL = os.environ.get("RUBRIC_MODEL", "GLM-5.3")

POOL_PROMPT = """You are creating grading criteria for a literature-synthesis question. A group of PhD-level experts wrote a reference answer using ONLY the reference passages below. Draft the ingredients (grading criteria) a correct, complete answer must contain.

Question: {question}

Reference passages (the source material the expert answer was written from):
{passages}

Reference (gold) answer written by the domain expert from these passages:
{gold}

Draft {n_drafts} alternative ingredient sets. Rules:
- 6-9 ingredients covering the distinct substantive points the answer needs
- two tiers: "critical" (core content, ~60-70% of total weight) and
  "valuable" (depth/enrichment, ~30-40%)
- each ingredient: name, criterion (one requirement sentence, self-contained),
  weight (float, all weights sum to 1.0), examples (2-5 short text snippets
  that WOULD satisfy the criterion — guidance, not required content)
- criteria must be answerable from the reference passages, not generic

Output JSON only: {{"drafts": [{{"ingredients": [{{"name": str, "tier": "critical"|"valuable", "criterion": str, "weight": float, "examples": [str, ...]}}, ...]}}, ...]}}"""

MERGE_PROMPT = """You are merging {n} draft ingredient sets for a literature-synthesis question into ONE final set. Dedupe overlapping criteria, keep the sharpest phrasing, preserve coverage, rebalance weights so they sum to exactly 1.0 (critical tier ~60-70%).

Question: {question}

Drafts (JSON):
{drafts}

Output the final set as JSON only: {{"ingredients": [{{"name": str, "tier": "critical"|"valuable", "criterion": str, "weight": float, "examples": [str, ...]}}, ...]}} with 6-10 ingredients."""


def _fmt_passages(ctxs: list, cap: int = 3) -> str:
    out = []
    for i, c in enumerate(ctxs[:cap]):
        t = c.get("text") if isinstance(c.get("text"), str) else "(no text)"
        out.append(f"[{i}] {c.get('title','')}\n{t[:1500]}")
    return "\n\n".join(out)


def gen_for_question(q: dict, n_drafts: int = 3) -> dict | None:
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    qid = q["id"]
    cache_p = os.path.join(DRAFTS_DIR, f"{qid}.json")
    if os.path.exists(cache_p):
        return json.load(open(cache_p, encoding="utf-8"))
    prompt = POOL_PROMPT.replace("{question}", q["input"]) \
        .replace("{passages}", _fmt_passages(q.get("ctxs") or [])) \
        .replace("{gold}", (q.get("output") or "")[:3000]) \
        .replace("{n_drafts}", str(n_drafts))
    drafts = None
    for temp, att in ((0.4, 0), (0.4, 1)):
        raw = call_paratera(prompt, model=MODEL, max_tokens=6000,
                            temperature=temp, enable_thinking=False) if att else \
            call_paratera(prompt, model=MODEL, max_tokens=6000, temperature=0.3,
                          enable_thinking=False)
        obj = parse_json_response(raw or "")
        if isinstance(obj, dict) and obj.get("drafts"):
            drafts = obj["drafts"]
            break
    if not drafts:
        return None
    # merge
    merge_p = MERGE_PROMPT.replace("{n}", str(len(drafts))) \
        .replace("{question}", q["input"]) \
        .replace("{drafts}", json.dumps(drafts, ensure_ascii=False)[:14000])
    final = None
    for _ in range(2):
        raw = call_paratera(merge_p, model=MODEL, max_tokens=6000, temperature=0.2,
                           enable_thinking=False)
        obj = parse_json_response(raw or "")
        if isinstance(obj, dict) and obj.get("ingredients"):
            final = obj["ingredients"]
            break
    if not final:
        return None
    # normalize weights + names (official convention: tiered names)
    tot = sum(float(i.get("weight") or 0) for i in final) or 1.0
    crit_n = val_n = 0
    for i in final:
        i["weight"] = round(float(i.get("weight") or 0) / tot, 6)
        tier = "critical" if str(i.get("tier")) == "critical" else "valuable"
        i["tier"] = tier
        idx = crit_n if tier == "critical" else val_n
        i["name"] = f"{'answer_critical' if tier == 'critical' else 'valuable'}_{idx}"
        if tier == "critical":
            crit_n += 1
        else:
            val_n += 1
    rec = {"question": q["input"], "case_id": qid, "annotator": "auto-glm53",
           "ingredients": final, "n_pooled_drafts": len(drafts)}
    json.dump(rec, open(cache_p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qs = load_questions(QFILE)
    if args.limit:
        qs = qs[:args.limit]
    # resume: keep already-generated
    done = {}
    if os.path.exists(OUT):
        for r in json.load(open(OUT, encoding="utf-8")):
            done[r["case_id"]] = r
    out = list(done.values())
    todo = [q for q in qs if q["id"] not in done]
    t0 = time.time()
    # parallel generation (2026-09-23): questions are independent; the serial
    # loop was an oversight (~45s/题 x 108 = 1.5h; 8 workers = ~12min)
    from concurrent.futures import ThreadPoolExecutor
    done_ids = {r["case_id"] for r in out}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("RUBRIC_GEN_WORKERS", "8"))) as ex:
        for rec in ex.map(gen_for_question, todo):
            if rec:
                out.append(rec)
                done_ids.add(rec["case_id"])
            n = len(done_ids)
            if n % 10 == 0:
                json.dump(out, open(OUT, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
                print(f"[gen] {n}/{len(qs)} rubrics "
                      f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n_ing = [len(r["ingredients"]) for r in out]
    print(f"[gen] DONE: {len(out)} rubrics, mean ingredients "
          f"{sum(n_ing) / max(1, len(n_ing)):.1f} -> {OUT}")


if __name__ == "__main__":
    main()
