# -*- coding: utf-8 -*-
"""SC runner CLI: drive the agent over a query list and write the official runs jsonl.

  python -m compilescholar.eval.sc.runner --bench-dir <shadow-bench> --pipeline sc-ours-shallow \
      --query-ids pilot50_ids.txt [--query-type core_query] [--index-dir <ephemeral>] [--workers 1] \
      [--max-llm-calls 8] [--evaluate]

  --bench-dir   the shadow bench (corpus/queries/rels + runs/ + results/); data/external is never written
  --index-dir   override the index root (tonight's ephemeral papers index; default = data/derived/index)
  --evaluate    after the run, score with the official evaluate.py (--only-run-queries) via subprocess

Per query: as_of = end of the source-paper month; the agent ranks pool candidates; the ranking is assembled
with the official segments (agent -> trajectory<=25 -> bm25 backfill, depth 100), the temporal filter and
source-paper withholding are applied deterministically, and the line is appended to
runs/agentic/<pipeline>/<query_type>.jsonl (resume: finished query_ids are skipped; agent failures go to
<query_type>.failed.jsonl and are retried on the next invocation).

External retrieval is switched off server-side (ToolConfig(external=False)) — closed-pool protocol. The arm is
"full" but never "summary": the summary arm would spend one LLM call per tool call.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import protocol
from .agent import AgentConfig, SCSearchAgent
from .backends import SystemSearch

_REPO = Path(__file__).resolve().parents[4]
_OFFICIAL_EVAL = _REPO / "third_party" / "scholarcatalyst" / "src" / "evaluation" / "evaluate.py"


def _configure_tools(index_dir: Path | None, budget_log: Path, run_id: str) -> None:
    from ...llm import client as llm
    llm.configure({}, run_id=run_id, ledger_path=budget_log.parent / "llm_calls.jsonl", caller="sc-runner")
    from ...tools import api
    api.configure(api.ToolConfig(arm="full", external=False, budget_log=str(budget_log)))
    from ...index.search import Index
    if index_dir is not None:
        ix = Index(embed=None, dir=index_dir)       # ephemeral BM25-only index (no embedding service needed)
    else:
        try:
            from ...llm.embedding import embed_local
            ix = Index(embed=embed_local)
        except Exception:                            # noqa: BLE001 — BM25-only fallback
            ix = Index(embed=None)
    opened = ix.warm()
    if opened["missing"]:
        print(f"[sc] index warm: missing {opened['missing']} (tools on those channels degrade to empty)")
    api.set_index(ix)


def _load_query_list(bench_dir: Path, qtypes: list[str], ids_file: str | None, limit: int | None):
    by_type = {qt: protocol.load_queries(bench_dir, qt) for qt in qtypes}
    if ids_file:
        wanted = {l.strip() for l in Path(ids_file).read_text(encoding="utf-8").splitlines() if l.strip()}
        found = {q["id"] for qs in by_type.values() for q in qs}
        missing = sorted(wanted - found)
        if missing:
            print(f"[sc] WARNING: {len(missing)} requested ids not in queries.jsonl: {missing[:5]}")
        by_type = {qt: [q for q in qs if q["id"] in wanted] for qt, qs in by_type.items()}
    if limit:
        # structural cut only (file order within the selected types) — never content-based
        n, out = 0, {}
        for qt in qtypes:
            keep = []
            for q in by_type[qt]:
                if n >= limit:
                    break
                keep.append(q)
                n += 1
            out[qt] = keep
        by_type = out
    return by_type


def run(bench_dir: Path, pipeline: str, qtypes: list[str], ids_file: str | None, limit: int | None,
        index_dir: Path | None, cfg: AgentConfig, workers: int, evaluate: bool,
        deep_read_cap: int = 0) -> dict:
    bench_dir = Path(bench_dir)
    out_root = bench_dir / "runs" / "agentic" / pipeline
    out_root.mkdir(parents=True, exist_ok=True)
    _configure_tools(index_dir, out_root / "tool_budget.jsonl",
                     run_id=f"sc-{pipeline}-{time.strftime('%Y%m%dT%H%M%S')}")

    corpus = protocol.Corpus(bench_dir)
    idmap = protocol.IdMap()
    by_type = _load_query_list(bench_dir, qtypes, ids_file, limit)
    n_sel = sum(len(v) for v in by_type.values())
    print(f"[sc] pipeline={pipeline} selected={n_sel} " +
          " ".join(f"{qt}={len(v)}" for qt, v in by_type.items()) + f" workers={workers} model={cfg.model}")
    if deep_read_cap:
        print(f"[sc] NOTE: deep_read_cap={deep_read_cap} requested — runtime deep extraction is not wired "
              f"into the shallow agent yet (full-form iteration item)")

    lock = threading.Lock()
    totals = {"run": 0, "skipped": 0, "failed": 0, "llm_calls": 0, "t0": time.monotonic()}

    for qt in qtypes:
        queries = by_type.get(qt) or []
        if not queries:
            continue
        backfill = protocol.load_backfill(bench_dir, qt)
        run_path = protocol.run_path(bench_dir, pipeline, qt)
        fail_path = protocol.failed_path(bench_dir, pipeline, qt)
        done = protocol.done_query_ids(run_path)
        pending = [q for q in queries if q["id"] not in done]
        totals["skipped"] += len(queries) - len(pending)

        manifest = {"pipeline": pipeline, "query_type": qt, "model": cfg.model, "backend": "system",
                    "index_dir": str(index_dir) if index_dir else "data/derived/index",
                    "agent_cfg": {k: v for k, v in cfg.__dict__.items()},
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (out_root / "run.json").write_text(
            json.dumps(json.loads((out_root / "run.json").read_text()) + [manifest]
                       if (out_root / "run.json").exists() else [manifest], indent=1))

        def one(q: dict) -> None:
            qid = q["id"]
            t0 = time.monotonic()
            as_of = protocol.as_of_for(q.get("paper_published", ""))
            withheld = protocol.withhold_ids(q, idmap)
            agent = SCSearchAgent(SystemSearch(idmap, corpus), cfg)   # per query: the trace never races
            try:
                res = agent.run(q, as_of, withheld)
            except Exception as e:                   # noqa: BLE001 — one bad query must not stop the run
                with lock:
                    totals["failed"] += 1
                    with open(fail_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"query_id": qid,
                                            "error": f"{type(e).__name__}: {str(e)[:300]}"}) + "\n")
                    print(f"  [{qt}] {qid}: FAILED ({type(e).__name__}: {str(e)[:120]})")
                return
            ranking = protocol.assemble(res.agent_ids, res.seen, backfill.get(qid, []),
                                        known_ids=corpus, withheld=withheld,
                                        query_date=q.get("paper_published", ""),
                                        corpus_dates=corpus.dates)
            segments = {s: sum(1 for r in ranking if r["source"] == s)
                        for s in ("agent", "trajectory", "backfill")}
            line = json.dumps({"query_id": qid, "ranking": ranking, "seen": res.seen, "segments": segments,
                               "summary": {**res.summary, "as_of": as_of, "withheld": sorted(withheld),
                                           "elapsed_s": round(time.monotonic() - t0, 1)}})
            with lock:
                totals["run"] += 1
                totals["llm_calls"] += res.summary.get("n_llm_calls", 0)
                with open(run_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                print(f"  [{qt}] {qid}: agent={segments['agent']} traj={segments['trajectory']} "
                      f"backfill={segments['backfill']} kept={res.summary['n_kept']} "
                      f"calls={res.summary['n_llm_calls']} ({time.monotonic() - t0:.0f}s) "
                      f"[{totals['run']}/{len(pending)}]", flush=True)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(one, pending))
        else:
            for q in pending:
                one(q)

    totals["minutes"] = round((time.monotonic() - totals["t0"]) / 60, 1)
    print(f"[sc] done: {totals['run']} run, {totals['skipped']} skipped(resume), {totals['failed']} failed, "
          f"{totals['llm_calls']} llm calls, {totals['minutes']} min")
    corpus.close()

    if evaluate and totals["run"]:
        env = {**os.environ, "BENCH_DIR": str(bench_dir), "PYTHONUTF8": "1"}
        cmd = [sys.executable, str(_OFFICIAL_EVAL), "--model", f"agentic/{pipeline}",
               "--bench-dir", str(bench_dir), "--only-run-queries"]
        print(f"[sc] official evaluate: {' '.join(cmd)}")
        subprocess.run(cmd, env=env, check=False)
    return totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--pipeline", required=True, help="run name under runs/agentic/")
    ap.add_argument("--query-type", default=None, choices=list(protocol.QUERY_TYPES),
                    help="default: both types present in the id list")
    ap.add_argument("--query-ids", default=None, help="file of query ids, one per line")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--index-dir", type=Path, default=None,
                    help="index root override (an ephemeral papers index while data/derived/index rebuilds)")
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--max-llm-calls", type=int, default=8)
    ap.add_argument("--per-query-k", type=int, default=25)
    ap.add_argument("--pool-cap", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--deep-read-cap", type=int, default=0,
                    help="full-form switch (runtime deep_read per query); 0 = off (shallow pilot)")
    ap.add_argument("--evaluate", action="store_true", help="score with the official evaluate.py afterwards")
    a = ap.parse_args()

    qtypes = [a.query_type] if a.query_type else list(protocol.QUERY_TYPES)
    cfg = AgentConfig(model=a.model, max_llm_calls=a.max_llm_calls, per_query_k=a.per_query_k,
                      pool_cap=a.pool_cap)
    run(a.bench_dir, a.pipeline, qtypes, a.query_ids, a.limit, a.index_dir, cfg, a.workers, a.evaluate,
        deep_read_cap=a.deep_read_cap)


if __name__ == "__main__":
    main()
