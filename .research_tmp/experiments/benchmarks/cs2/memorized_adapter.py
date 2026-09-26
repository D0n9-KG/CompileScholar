# -*- coding: utf-8 -*-
"""Memorized baselines -> official CS2 format adapter + GLM re-judging prep.

Three baselines (from allenai/asta-bench-solver-data):
  Elicit      responses.json  — already in sections format (direct)
  Perplexity  openai_dr/*.json — markdown report (needs sections wrapping)
  SciSpace    scispace/test.json — response text (needs sections wrapping)

The wrapping follows the official format_solver convention (single
section, no citations — these systems don't emit citation JSON;
citation facets score via the is_retrieverless 0.5-title-tier path,
disclosed).

Output: arm_memorized/answers_{system}_{split}.json — rows keyed by
question text matching the rubrics, ready for the official scorer.
"""
import json
import sys
from pathlib import Path

CS2 = Path(__file__).resolve().parent
MEM = CS2 / "arm_memorized"
RUBRICS = CS2.parent / "scholarqa_multi" / "sqa2_rubrics_v1_recomputed.json"


def load_rubrics(split="dev"):
    version = "v1" if split == "dev" else "v2"
    p = CS2.parent / "scholarqa_multi" / f"sqa2_rubrics_{version}_recomputed.json"
    return json.load(open(p, encoding="utf-8"))


def adapt_elicit(split="dev"):
    """Elicit responses.json: rows are {id, sections} — id is a topic
    slug, not the case_id. Match to rubrics by ... we can't match by id;
    the dataset covers both splits (201 rows). For dev judging we need
    the question mapping — Elicit rows are ordered by their original
    evaluation; absent a key, match by content overlap is unreliable.
    PRACTICAL route: judge the FULL Elicit set against whichever rubric
    question the sections answer — the official scorer needs
    question+response pairs. We emit Elicit rows paired with their
    best-matching rubric question via title-token overlap.
    """
    data = json.load(open(MEM / "sqa_elicit_responses.json",
                          encoding="utf-8"))
    rubrics = load_rubrics(split)
    out = []
    for r in data:
        if not r.get("sections"):
            continue
        # title-slug -> question matching (token overlap)
        slug = (r.get("id") or "").lower()
        best, best_score = None, 0
        for q in rubrics:
            qtoks = {t for t in q["question"].lower().split() if len(t) > 3}
            stoks = {t for t in slug.replace("-", " ").split()}
            ov = len(qtoks & stoks)
            if ov > best_score:
                best, best_score = q, ov
        if best is None or best_score < 2:
            continue
        out.append({"qid": best["case_id"][:24],
                    "question": best["question"],
                    "sections": r["sections"],
                    "source": "elicit",
                    "match_score": best_score})
    return out


def adapt_text_system(name, fname, split, answer_field):
    """Perplexity/SciSpace: {question, answer/response} rows — match
    question text to rubrics (exact/normalized)."""
    data = json.load(open(MEM / fname, encoding="utf-8"))
    rubrics = load_rubrics(split)
    qmap = {}
    for q in rubrics:
        qmap[_norm(q["question"])] = q
    out = []
    for r in data:
        q = qmap.get(_norm(r.get("question") or ""))
        if not q:
            continue
        text = r.get(answer_field) or ""
        if not text:
            continue
        # markdown report -> single section (official format_solver
        # convention: text systems emit one body section, no citations)
        out.append({"qid": q["case_id"][:24],
                    "question": q["question"],
                    "sections": [{"title": "Report", "text": text,
                                  "citations": []}],
                    "source": name})
    return out


def _norm(s):
    import re
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = []
    # Perplexity dev
    rows = adapt_text_system("perplexity_dr",
                             "sqa_openai_dr_dev.json", "dev", "answer")
    print(f"perplexity dev matched: {len(rows)}/100")
    json.dump(rows, open(MEM / "answers_perplexity_dr_dev.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    # SciSpace 只有 test——dev20 用不了，记录
    # Elicit
    rows = adapt_elicit("dev")
    print(f"elicit dev matched: {len(rows)}")
    json.dump(rows, open(MEM / "answers_elicit_dev.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
