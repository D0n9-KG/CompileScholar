# -*- coding: utf-8 -*-
"""DeepScholar-Bench nugget judging and summary (moved from review_1002/p6/{judge_nuggets,analyze}.py).

Judging calls the official Nuggetizer.assign unchanged (same prompt, window 10, model, temperature 0) and keeps the
per-nugget assignments. nugget_coverage = strict_all_score (support only), the official metric.

Change from the original (W1-5, intentional): a question whose judging fails is recorded as an error row instead of
being skipped, and `summarize` scores missing answers / failed questions as 0 over the full question set (the
original analysis took the intersection of answered questions — the opposite of the CS2 convention).
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..core import paths, secrets

MODEL = "DeepSeek-V4.1-Flash"


def dsb_root() -> Path:
    return Path(os.environ.get("CS_DSB") or paths.legacy_bench() / "deepscholar" / "dsb")


_lock = threading.Lock()
_local = threading.local()
CALLS = {"n": 0, "err": 0}


def _nuggetizer():
    if getattr(_local, "nz", None) is None:
        src = str(dsb_root() / "eval" / "nuggetizer" / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        os.environ["OPENAI_API_KEY"] = secrets.get("PARATERA_API_KEY", "")
        os.environ["OPENAI_API_BASE"] = secrets.get("PARATERA_BASE_URL", "")
        from nuggetizer.models.nuggetizer import Nuggetizer
        nz = Nuggetizer(model=MODEL, log_level=0)
        orig = nz.assigner_llm.client.chat.completions.create

        def counted(*a, **k):
            with _lock:
                CALLS["n"] += 1
            try:
                return orig(*a, **k)
            except Exception:
                with _lock:
                    CALLS["err"] += 1
                raise
        nz.assigner_llm.client.chat.completions.create = counted
        _local.nz = nz
    return _local.nz


def judge_one(gt_dir: str, text: str) -> dict:
    from nuggetizer.core.metrics import calculate_nugget_scores
    from nuggetizer.core.types import ScoredNugget
    gt = json.load(open(dsb_root() / "dataset" / "gt_nuggets_outputs" / gt_dir / "res.json", encoding="utf-8"))
    nuggets = [ScoredNugget(text=n["text"], importance=n.get("importance", "vital")) for n in gt.get("supported_nuggets", [])]
    t0 = time.time()
    assigned = _nuggetizer().assign(gt.get("query", ""), text or "", nuggets)
    nl = [{"text": n.text, "importance": n.importance, "assignment": n.assignment} for n in assigned]
    m = calculate_nugget_scores(gt_dir, nl)
    return {"gt_dir": gt_dir, "qid": gt["qid"], "nugget_coverage": m.strict_all_score, "strict_vital": m.strict_vital_score,
            "all_partial": m.all_score, "vital_partial": m.vital_score, "n_nuggets": len(nl),
            "n_failed": sum(1 for x in nl if x["assignment"] == "failed"), "words": len((text or "").split()),
            "judge_s": round(time.time() - t0, 1), "nuggets": nl}


def judge_dir(texts_dir: str, out_path: str, ids: list[str] | None = None, workers: int = 6) -> dict:
    """Judge <texts_dir>/<gt_dir>.md for every id; resumable; failures become {"gt_dir", "error"} rows (retried on
    the next call)."""
    ids = ids or sorted([f[:-3] for f in os.listdir(texts_dir) if f.endswith(".md") and f[:-3].isdigit()], key=int)
    done = {r["gt_dir"]: r for r in json.load(open(out_path, encoding="utf-8"))} if os.path.exists(out_path) else {}
    done = {k: v for k, v in done.items() if "error" not in v}
    todo = [i for i in ids if i not in done]
    _nuggetizer()

    def _save():
        json.dump(sorted(done.values(), key=lambda x: int(x["gt_dir"])), open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_one, i, open(os.path.join(texts_dir, i + ".md"), encoding="utf-8").read()): i for i in todo}
        for f in as_completed(futs):
            i = futs[f]
            try:
                r = f.result()
            except Exception as e:
                r = {"gt_dir": i, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            with _lock:
                done[i] = r
                _save()
    return {"n": len(done), "error_rows": sum(1 for v in done.values() if "error" in v), "calls": CALLS["n"],
            "call_errors": CALLS["err"]}


def question_set() -> list[str]:
    """The 48 unique DSB questions (smallest gt_dir per qid), as used by every arm."""
    rows = json.load(open(paths.REPO / ".research_tmp" / "review_1002" / "p6" / "oracle_inputs.json", encoding="utf-8"))
    return [r["gt_dir"] for r in rows]


def per_question(judged_path: str, qset: list[str]) -> dict[str, float]:
    """nugget_coverage per question over the full set; missing / error rows count 0."""
    rows = {r["gt_dir"]: r for r in json.load(open(judged_path, encoding="utf-8"))} if os.path.exists(judged_path) else {}
    return {q: (rows[q]["nugget_coverage"] if q in rows and "error" not in rows[q] else 0.0) for q in qset}


def summarize(judged_path: str, qset: list[str]) -> dict:
    pq = per_question(judged_path, qset)
    rows = {r["gt_dir"]: r for r in json.load(open(judged_path, encoding="utf-8"))} if os.path.exists(judged_path) else {}
    missing = [q for q in qset if q not in rows or "error" in rows[q]]
    return {"n": len(qset), "n_missing_as_zero": len(missing), "missing": missing,
            "nugget_coverage": sum(pq.values()) / len(qset) if qset else None}


def paired(a_path: str, b_path: str, qset: list[str], B: int = 10000, seed: int = 0) -> dict:
    a, b = per_question(a_path, qset), per_question(b_path, qset)
    d = [a[q] - b[q] for q in qset]
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(B))
    return {"mean": sum(d) / len(d), "ci": (bs[int(0.025 * B)], bs[int(0.975 * B) - 1]),
            "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d)}
