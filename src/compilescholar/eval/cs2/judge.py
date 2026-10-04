# -*- coding: utf-8 -*-
"""CS2 judging with the official astabench scorers, driven directly (moved from cs2/direct_judge.py).

The scorer functions (score_sqa / score_precision / score_citation) and their arguments are the official ones
(task.py score_all defaults: simplified_eval, assess_jointly, all_at_once, temperature 0.5, top_p 0.95); only the
event loop around them is ours (inspect's eval scheduler hung repeatedly). Local deviations from the official
pipeline are collected in JudgeAdapter, each switchable and recorded in every output row:

  connection_close  : force `Connection: close` on httpx (Paratera keep-alive hang). Network only, no effect on scores.
  max_retries_4     : generate_with_retry max_retries 20 -> 4, base_delay 1.0 (fewer retries on malformed JSON).
  idx_zero_to_one   : shift 0-based criteria_idx to 1-based (DeepSeek numbers from 0; validator expects 1).
  judge_model       : DeepSeek-V4.1-Flash via Paratera instead of the official google/gemini-3-flash-preview.

`JudgeAdapter.legacy()` reproduces every frozen v9b number; `JudgeAdapter.official()` keeps only the network fix and
uses the official judge model through OpenRouter.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass

from ...core import paths, secrets
from .scoring import RUBRIC_FILES, rubric_path

# official scorer model (astabench task.py:406,468), reached through inspect's native OpenRouter provider
OFFICIAL_JUDGE = "openrouter/google/gemini-3-flash-preview"
LEGACY_JUDGE = "openai-api/paratera/DeepSeek-V4.1-Flash"


@dataclass(frozen=True)
class JudgeAdapter:
    connection_close: bool = True
    max_retries_4: bool = True
    idx_zero_to_one: bool = True
    judge_model: str = LEGACY_JUDGE

    @classmethod
    def legacy(cls) -> "JudgeAdapter":
        return cls()

    @classmethod
    def official(cls) -> "JudgeAdapter":
        return cls(connection_close=True, max_retries_4=False, idx_zero_to_one=False, judge_model=OFFICIAL_JUDGE)

    @classmethod
    def primary(cls) -> "JudgeAdapter":
        """Primary measure for upgraded-system runs (PREREG amendment 1): DeepSeek-V4.1-Flash, official retry count;
        keeps only the two format/network fixes that do not change scoring (Connection: close; 0-based criteria
        indices that DeepSeek emits renumbered to the 1-based form the official validator expects)."""
        return cls(connection_close=True, max_retries_4=False, idx_zero_to_one=True, judge_model=LEGACY_JUDGE)


def astabench_path() -> str:
    """astabench is a pinned third-party checkout (editable install); CS_ASTABENCH overrides the location."""
    return os.environ.get("CS_ASTABENCH") or str(paths.REPO / ".research_tmp" / "scratch" / "ai2_baseline_2026-09-08" /
                                                 "asta" / "asta-bench-main")


def _provider_env():
    """inspect's openai-api/<service>/<model> provider reads <SERVICE>_API_KEY / <SERVICE>_BASE_URL."""
    os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder")
    pk, pb = secrets.get("PARATERA_API_KEY", ""), secrets.get("PARATERA_BASE_URL", "")
    for svc in ("PARATERA", "DEEPSEEK", "GLM"):
        os.environ[f"{svc}_API_KEY"] = pk
        os.environ[f"{svc}_BASE_URL"] = pb
    ok = secrets.get("OPENROUTER_API_KEY")
    if ok:
        os.environ["OPENROUTER_API_KEY"] = ok


_INSTALLED: JudgeAdapter | None = None


def install(adapter: JudgeAdapter):
    """Apply the adapter's patches once per process (they are module-level monkeypatches on httpx / astabench)."""
    global _INSTALLED
    if _INSTALLED is not None:
        if _INSTALLED != adapter:
            raise RuntimeError("a different JudgeAdapter is already installed in this process")
        return
    if astabench_path() not in sys.path:
        sys.path.insert(0, astabench_path())
    _provider_env()
    if adapter.connection_close:
        import httpx
        orig_send, orig_asend = httpx.Client.send, httpx.AsyncClient.send

        def _send(self, request, **kw):
            request.headers["Connection"] = "close"
            return orig_send(self, request, **kw)

        async def _asend(self, request, **kw):
            request.headers["Connection"] = "close"
            return await orig_asend(self, request, **kw)
        httpx.Client.send, httpx.AsyncClient.send = _send, _asend
    if adapter.max_retries_4 or adapter.idx_zero_to_one:
        import astabench.evals.sqa.citation_eval as m_cit
        import astabench.evals.sqa.precision_eval as m_prec
        import astabench.evals.sqa.retry_utils as ru
        import astabench.evals.sqa.rubric as m_rub
        orig = ru.generate_with_retry

        async def _gwr(*a, **kw):
            if adapter.max_retries_4:
                kw.setdefault("max_retries", 4)
                kw.setdefault("base_delay", 1.0)
            if adapter.idx_zero_to_one and kw.get("parsed_validator") is not None:
                ov = kw["parsed_validator"]
                kw["parsed_validator"] = lambda parsed, _ov=ov: _ov(_zero_to_one(parsed))
            return await orig(*a, **kw)
        for m in (m_rub, m_prec, m_cit):
            m.generate_with_retry = _gwr
    _INSTALLED = adapter


def _zero_to_one(parsed):
    """0-based -> 1-based criteria_idx (only when 0 is present and the indices fit the 1-based range)."""
    try:
        scores = parsed.get("scores")
        if isinstance(scores, list) and scores:
            idxs = [s.get("criteria_idx") for s in scores if isinstance(s, dict) and isinstance(s.get("criteria_idx"), int)]
            if idxs and 0 in idxs and max(idxs) <= len(scores):
                for s in scores:
                    if isinstance(s, dict) and isinstance(s.get("criteria_idx"), int):
                        s["criteria_idx"] += 1
    except Exception:
        pass
    return parsed


class _Out:
    def __init__(self, completion):
        self.completion = completion


class _State:
    def __init__(self, completion, metadata):
        self.output = _Out(completion)
        self.metadata = metadata


SCORER_CAP = 12


async def judge_one(question: str, rubric_row: dict, report_json: str, adapter: JudgeAdapter, sem) -> dict:
    from inspect_ai.model import get_model
    from inspect_ai.scorer import Target
    from astabench.evals.sqa.task import score_citation, score_precision, score_sqa
    model = get_model(adapter.judge_model)
    # openai-api/<svc>/<model>: that provider sends '<svc>/<model>' as the model name; the service wants the bare
    # name. Native providers (openrouter/...) need no rewrite.
    if adapter.judge_model.startswith("openai-api/") and adapter.judge_model.count("/") >= 2:
        short = adapter.judge_model.split("/", 2)[-1]
        if hasattr(model, "api") and hasattr(model.api, "model_name"):
            model.api.model_name = short
    target = Target([json.dumps(rubric_row, indent=2)])
    scorers = [("ingredient_recall", score_sqa(model, simplified_eval=True, assess_jointly=True, temperature=0.5, top_p=0.95)),
               ("answer_precision", score_precision(model)),
               ("citation", score_citation(model, sentence_wise_cit_eval=False, all_at_once=True, temperature=0.5, top_p=0.95))]

    async def _run(name, sc):
        async with sem:
            s = await sc(_State(report_json, {"initial_prompt": question}), target)
        v = s.value if isinstance(s.value, dict) else {"score": s.value}
        md = getattr(s, "metadata", None)
        if md:
            try:
                v["_meta"] = json.loads(json.dumps(md, default=str))
            except Exception:
                v["_meta_raw"] = str(md)[:4000]
        ex = getattr(s, "explanation", None)
        if ex:
            v["_explanation"] = str(ex)[:2000]
        return name, v
    pairs = await asyncio.gather(*[_run(n, sc) for n, sc in scorers], return_exceptions=True)
    out: dict = {}
    for p in pairs:
        if isinstance(p, Exception):
            out.setdefault("_errors", []).append(f"{type(p).__name__}: {str(p)[:150]}")
        else:
            out[p[0]] = p[1]
    return out


def _rubric_sha(split: str) -> str:
    return hashlib.sha256(open(rubric_path(split), "rb").read()).hexdigest()


async def judge_file(in_path: str, out_path: str, split: str, adapter: JudgeAdapter, parallel: int = 6,
                     quiet_scores: bool | None = None) -> dict:
    """Judge every row of a judge-input file (qid, question, sections). Resumable: rows already judged are kept;
    rows with judge errors are retried. Every row records judge model, adapter switches, rubric hash and time.
    quiet_scores (default: True for split=test) suppresses per-question score printing."""
    install(adapter)
    if split not in RUBRIC_FILES:
        raise ValueError(f"split must be one of {sorted(RUBRIC_FILES)}")
    quiet = (split == "test") if quiet_scores is None else quiet_scores
    rows = json.load(open(in_path, encoding="utf-8"))
    rmap = {q["case_id"][:24]: q for q in json.load(open(rubric_path(split), encoding="utf-8"))}
    done = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    retry = [q for q, v in done.items() if isinstance(v, dict) and ("error" in v or "_errors" in v)]
    for q in retry:
        done.pop(q)
    todo = [r for r in rows if r["qid"] not in done]
    print(f"[judge] split={split} judge={adapter.judge_model} todo={len(todo)} resumed={len(done)} retry={len(retry)}",
          flush=True)
    stamp = {"judge_model": adapter.judge_model, "adapter": asdict(adapter), "rubric_sha256": _rubric_sha(split),
             "split": split}
    sem_q, sem_s, lock = asyncio.Semaphore(parallel), asyncio.Semaphore(SCORER_CAP), asyncio.Lock()

    async def _one(r):
        report = json.dumps({"sections": r["sections"]}, ensure_ascii=False)
        async with sem_q:
            try:
                scores = await judge_one(r["question"], rmap[r["qid"]], report, adapter, sem_s)
            except Exception as e:
                scores = {"error": str(e)[:200]}
            scores["_judge"] = {**stamp, "time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
            async with lock:
                done[r["qid"]] = scores
                json.dump(done, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            err = "error" in scores or "_errors" in scores
            print(f"  judged {r['qid'][:14]}" + (" ERROR" if err else "") + ("" if quiet else f" {_flat(scores)}"),
                  flush=True)
    await asyncio.gather(*[_one(r) for r in todo])
    n_err = sum(1 for v in done.values() if isinstance(v, dict) and ("error" in v or "_errors" in v))
    return {"n": len(done), "judge_error_rows": n_err}


def _flat(scores: dict) -> dict:
    out = {}
    for k, v in scores.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            nums = [vv for vv in v.values() if isinstance(vv, (int, float))]
            out[k] = nums[0] if nums else None
    return out
