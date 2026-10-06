# -*- coding: utf-8 -*-
"""compilescholar command line.

  compilescholar build  status | all | papers | documents | citations | extract | index   (literature layer)
  compilescholar answer --config configs/bench/cs2_dev_confirm.yaml --run-id cs2-dev-vnext-20261005-x [--set k=v ...]
                        (--set answer.mode=lit answers through the literature-layer tools)
  compilescholar judge  --run-id <id> [--legacy]
  compilescholar score  --split test --ref runs/<a>/scores.json,runs/<b>/scores.json --others harness=<file> ...
  compilescholar verify --run-id <id>

Every run writes runs/<run_id>/{config.resolved.yaml, manifest.json, answers.json, judge_input.json[, scores.json]}.
A run directory is resumable only with the same resolved configuration (different config -> refused).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import yaml

from .core import config as C
from .core import manifest as M
from .core import paths


def _run_dir(run_id: str) -> Path:
    return paths.runs() / run_id


def _questions(cfg: C.RunConfig) -> tuple[list[dict], Path]:
    if cfg.bench.name != "cs2":
        raise SystemExit(f"bench {cfg.bench.name!r}: only cs2 is wired into the CLI so far")
    from .eval.cs2.scoring import rubric_path
    p = Path(rubric_path(cfg.bench.split))
    qs = json.load(open(p, encoding="utf-8"))[cfg.bench.offset:cfg.bench.offset + cfg.bench.limit]
    return [{"qid": q["case_id"][:24], "question": q["question"]} for q in qs], p


def cmd_answer(a):
    cfg = C.load(a.config, a.set)
    rd = _run_dir(a.run_id)
    resolved = yaml.safe_dump(cfg.to_dict(), sort_keys=True, allow_unicode=True)
    cp = rd / "config.resolved.yaml"
    if cp.exists() and cp.read_text(encoding="utf-8") != resolved:
        raise SystemExit(f"{rd} exists with a different configuration; use a new --run-id")
    rd.mkdir(parents=True, exist_ok=True)
    cp.write_text(resolved, encoding="utf-8")
    # the LLM client is configured from the resolved config (runtime.local_max_concurrent is the answer path's lane
    # width, kept for the recorded v9b configs); its ledger goes into the run directory
    from .llm import client as LC
    LC.configure(_llm_settings(cfg, local_lanes=cfg.runtime.local_max_concurrent), run_id=a.run_id,
                 ledger_path=rd / "llm_calls.jsonl", caller="answer")
    os.environ["SCIVERSE_MAX_WAIT_S"] = str(cfg.runtime.sciverse_max_wait_s)
    if cfg.answer.cutoff:
        os.environ["KNOWLEDGE_CUTOFF"] = str(cfg.answer.cutoff)
    os.environ["ANSWER_MODEL"] = cfg.answer.model
    from .answer import pipeline as AP
    qs, qfile = _questions(cfg)
    kbp = cfg.kb_path()
    inputs = {"questions": qfile}
    if kbp:
        inputs.update({f"kb/{n}": kbp / n for n in ("papers.json", "records.json", "record_vecs.f32",
                                                     "record_vecs.meta.json", "state_merged.json")})
    M.write(rd, M.build(a.run_id, cfg.to_dict(), inputs))
    ans_p = rd / "answers.json"
    done = {r["qid"]: r for r in json.load(open(ans_p, encoding="utf-8"))} if ans_p.exists() else {}
    if a.retry_errors:
        done = {k: v for k, v in done.items() if not v.get("error")}
    lit = cfg.answer.mode == "lit"
    if lit:
        from .dfc import store as _st
        for s in ("papers", "documents", "citations", "extract", "index"):
            inputs[f"dfc/{s}.manifest"] = _st.root() / "manifests" / f"{s}.json"
        _st.require_fresh("papers", "documents", "citations", "extract", "index")
        M.write(rd, M.build(a.run_id, cfg.to_dict(), inputs))
    kb = AP.KB(str(kbp)) if kbp and not lit else None
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[answer] {a.run_id}: {len(todo)} to answer, {len(done)} done (mode={cfg.answer.mode})", flush=True)

    def one(q):
        t = time.time()
        try:
            if lit:
                r = AP.answer_lit(q["question"], cutoff=cfg.answer.cutoff, use_screen=cfg.answer.screen,
                                  word_budget=cfg.answer.word_budget)
            else:
                r = AP.answer(q["question"], kb, use_ext=cfg.answer.ext, use_cite=cfg.answer.cite,
                              use_screen=cfg.answer.screen, use_state=cfg.answer.state, use_probe=cfg.answer.probe,
                              word_budget=cfg.answer.word_budget, cutoff=cfg.answer.cutoff)
            r.update(q)
        except Exception as e:
            r = {**q, "sections": [], "error": f"{type(e).__name__}: {str(e)[:300]}"}
        tr = r.get("trace") or {}
        print(f"  {q['qid']} {time.time() - t:.0f}s words={tr.get('words')} ev={tr.get('by_src')} err={r.get('error')}",
              flush=True)
        return r
    with cf.ThreadPoolExecutor(cfg.runtime.workers) as ex:
        for r in ex.map(one, todo):
            done[r["qid"]] = r
            tmp = ans_p.with_suffix(".tmp")
            json.dump(list(done.values()), open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, ans_p)
    rows = [{"qid": r["qid"], "question": r["question"], "sections": r.get("sections") or []} for r in done.values()]
    json.dump(rows, open(rd / "judge_input.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[answer] wrote {len(rows)} rows -> {rd}", flush=True)


def cmd_judge(a):
    import asyncio
    from .eval.cs2 import judge as J
    rd = _run_dir(a.run_id)
    cfg = C.load(rd / "config.resolved.yaml")
    # default = primary measure (PREREG amendment 1); --legacy reproduces the v9b judging setup exactly
    adapter = J.JudgeAdapter.legacy() if a.legacy else J.JudgeAdapter.primary()
    out = rd / ("scores_legacy.json" if a.legacy else "scores.json")
    res = asyncio.run(J.judge_file(str(rd / "judge_input.json"), str(out), cfg.bench.split, adapter, parallel=a.parallel))
    print(f"[judge] {res}", flush=True)
    if res["judge_error_rows"]:
        raise SystemExit("judge error rows remain: re-run `judge` until 0 (they are never counted as 0)")


def cmd_score(a):
    sys.argv = ["paired_stats", a.split, "--ref", a.ref, "--others", *a.others]
    from .eval import stats
    stats.main()


def cmd_verify(a):
    rd = _run_dir(a.run_id)
    cfg = C.load(rd / "config.resolved.yaml")
    _, qfile = _questions(cfg)
    kbp = cfg.kb_path()
    inputs = {"questions": qfile}
    if kbp:
        inputs.update({f"kb/{n}": kbp / n for n in ("papers.json", "records.json", "record_vecs.f32",
                                                     "record_vecs.meta.json", "state_merged.json")})
    r = M.verify(rd / "manifest.json", inputs)
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["ok"] else 1)


def _llm_settings(cfg, local_lanes: int | None = None) -> dict:
    llm = json.loads(json.dumps(cfg.llm or {}))
    if local_lanes:
        llm.setdefault("providers", {}).setdefault("local", {})["lanes"] = int(local_lanes)
    return llm


def cmd_build(a):
    """Build one stage of the literature layer (or report the status of every stage). Each stage refuses to run on a
    stale upstream, so stages are built in order; `all` builds every stale stage in order."""
    from .dfc import store
    if a.stage == "status":
        for s in store.STAGES:
            st = store.status(s)
            m = store.read_manifest(s) or {}
            print(f"{s:10s} built={st['built']} stale={st['stale']} behind={st['behind']} complete={st['complete']} "
                  f"counts={m.get('counts')} work={m.get('work')} why={st['why']}")
        return
    order = store.STAGES
    if a.stage == "all":
        todo = [s for s in order if (lambda st: not st["built"] or st["stale"] or st["behind"] or not st["complete"])(
            store.status(s))]
        # once one stage is rebuilt every stage after it must follow
        todo = list(order[order.index(todo[0]):]) if todo else []
    else:
        todo = [a.stage]
    rb = a.rebuild
    cfg = C.load(a.config, a.set)
    from .llm import client as LC
    run_id = f"build-{'-'.join(todo) or 'none'}-{time.strftime('%Y%m%dT%H%M%S')}"
    LC.configure(_llm_settings(cfg), run_id=run_id, caller="build")
    print(f"[build] stages {todo}; LLM ledger {LC.call_log_summary()['ledger']}", flush=True)
    for s in todo:
        print(f"[build] {s}", flush=True)
        if s == "papers":
            from .corpus import papers
            print(json.dumps(papers.build()))
        elif s == "documents":
            from .documents import build as B
            print(json.dumps(B.build(rebuild=rb)))
        elif s == "citations":
            from .citations import build as B
            print(json.dumps(B.build(rebuild=rb)))
        elif s == "extract":
            from .extract import build as B
            bc = cfg.build
            cats = tuple(a.categories.split(",")) if a.categories else tuple(bc.categories)
            print(json.dumps(B.build(categories=cats, since=a.since or bc.since,
                                     n_deep=a.n_deep if a.n_deep is not None else bc.n_deep,
                                     workers=a.workers or bc.workers, rebuild=rb)))
        elif s == "index":
            from .index import build as B
            dense = tuple(x for x in a.dense.split(",") if x) if a.dense is not None else tuple(cfg.build.dense)
            print(json.dumps(B.build(dense=dense)))


def cmd_library(a):
    """The registry (data/library/registry.sqlite): bulk imports, title-match proposals, counts."""
    from .library import identity, import_arxiv, import_benchmarks, import_scievo, store
    if a.action == "import-arxiv":
        print(json.dumps(import_arxiv.run(scope=a.scope, fill_members=a.fill_members)))
    elif a.action == "import-scievo":
        print(json.dumps(import_scievo.run()))
    elif a.action == "adjudicate":
        from .library import adjudicate
        from .llm import client as LC
        cfg = C.load(a.config, a.set)
        LC.configure(_llm_settings(cfg), run_id=f"library-adjudicate-{time.strftime('%Y%m%dT%H%M%S')}",
                     caller="library")
        kinds = tuple(a.kinds.split(",")) if a.kinds else ("shared_doi", "identifier_conflict", "title_match")
        only = [int(x) for x in open(a.ids).read().split()] if a.ids else None
        print(json.dumps(adjudicate.run(kinds=kinds, limit=a.limit, apply=not a.dry_run, only_ids=only,
                                        workers=a.workers or 32)))
    elif a.action == "import-benchmarks":
        names = tuple(a.benchmarks.split(",")) if a.benchmarks else tuple(import_benchmarks.BENCHMARKS)
        print(json.dumps(import_benchmarks.run(names)))
    elif a.action == "propose-titles":
        with store.lock():
            con = store.connect()
            print(json.dumps(identity.propose_title_matches(con)))
            con.close()
    elif a.action == "status":
        con = store.connect()
        q = lambda s: con.execute(s).fetchall()
        print(json.dumps({
            "papers": dict(q("SELECT status, count(*) FROM papers GROUP BY status")),
            "id_prefix": dict(q("SELECT substr(paper_id, 1, instr(paper_id, ':') - 1), count(*) FROM papers "
                                "WHERE status='active' GROUP BY 1")),
            "identifiers": dict(q("SELECT scheme, count(*) FROM identifiers GROUP BY scheme")),
            "members": dict(q("SELECT benchmark, count(*) FROM members GROUP BY benchmark")),
            "first_public": dict(q("SELECT coalesce(first_kind, 'none'), count(*) FROM papers WHERE status='active' "
                                   "GROUP BY 1")),
            "merge_queue": [list(r) for r in q("SELECT kind, status, count(*) FROM merge_queue GROUP BY 1, 2")]},
            indent=1))
        con.close()


def cmd_acquire(a):
    """Fetch and identity-check full texts for registry papers: a benchmark's members, or ids from a file."""
    import random
    from .acquire import run as R
    from .library import store
    from .llm import client as LC
    cfg = C.load(a.config, a.set)
    LC.configure(_llm_settings(cfg), run_id=f"acquire-{time.strftime('%Y%m%dT%H%M%S')}", caller="acquire")
    con = store.connect()
    if a.ids:
        pids = [x.strip() for x in open(a.ids, encoding="utf-8") if x.strip()]
    else:
        pids = [p for (p,) in con.execute("SELECT DISTINCT paper_id FROM members WHERE benchmark=?", (a.benchmark,))]
    con.close()
    if a.sample:
        pids = random.Random(a.seed).sample(sorted(pids), min(a.sample, len(pids)))
    exclude = tuple(x for x in (a.exclude or "").split(",") if x)
    print(json.dumps(R.acquire(pids, workers=a.workers, exclude=exclude, llm=not a.no_llm)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="compilescholar")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("acquire", help="fetch + identity-check full texts (library assets)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--benchmark", help="every member of this benchmark")
    g.add_argument("--ids", help="file of paper_ids, one per line")
    p.add_argument("--sample", type=int, default=None, help="a random sample of this size (seed --seed)")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--exclude", default="", help="channels not to use, e.g. scihub_local (CS main experiments)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--no-llm", action="store_true", help="unsure identity checks become mismatches")
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[])
    p.set_defaults(fn=cmd_acquire)
    p = sub.add_parser("library", help="the registry: import-arxiv | import-benchmarks | import-scievo | "
                                       "propose-titles | status")
    p.add_argument("action", choices=["import-arxiv", "import-benchmarks", "import-scievo", "propose-titles",
                                      "adjudicate", "status"])
    p.add_argument("--kinds", default=None, help="adjudicate: queue kinds (default all)")
    p.add_argument("--limit", type=int, default=None, help="adjudicate: at most this many pairs")
    p.add_argument("--dry-run", action="store_true", help="adjudicate: store verdicts, merge nothing")
    p.add_argument("--ids", default=None, help="adjudicate: file of merge_queue ids (an audit sample)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--fill-members", action="store_true",
                   help="import-arxiv: only the snapshot records of registry arXiv ids that lack version dates")
    p.add_argument("--scope", default="cs", choices=["cs", "all"], help="import-arxiv: cs (default) or every record")
    p.add_argument("--benchmarks", default=None, help="import-benchmarks: comma list (default all four)")
    p.set_defaults(fn=cmd_library)
    p = sub.add_parser("build", help="build a literature-layer stage: status | all | papers | documents | citations | "
                                     "extract | index")
    p.add_argument("stage", choices=["status", "all", "papers", "documents", "citations", "extract", "index"])
    p.add_argument("--config", help="experiment config (default: configs/base.yaml [+ local.yaml]); build: and llm:")
    p.add_argument("--set", action="append", default=[], help="override, e.g. build.n_deep=200")
    p.add_argument("--categories", default=None, help="extract: primary categories in scope (default build.categories)")
    p.add_argument("--since", default=None, help="extract: first v1 date in scope (default build.since)")
    p.add_argument("--n-deep", type=int, default=None, help="extract: number of T2 papers (default build.n_deep)")
    p.add_argument("--workers", type=int, default=None, help="default build.workers")
    p.add_argument("--dense", default=None, help="index: which indexes get dense vectors (default build.dense)")
    p.add_argument("--rebuild", action="store_true",
                   help="build into a new file and swap it in (clean slate); default is incremental")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("answer")
    p.add_argument("--config")
    p.add_argument("--run-id", required=True)
    p.add_argument("--set", action="append", default=[], help="override, e.g. answer.cite=false")
    p.add_argument("--retry-errors", action="store_true")
    p.set_defaults(fn=cmd_answer)
    p = sub.add_parser("judge")
    p.add_argument("--run-id", required=True)
    p.add_argument("--legacy", action="store_true", help="v9b judging setup (adds max_retries 4); default is the primary measure")
    p.add_argument("--parallel", type=int, default=6)
    p.set_defaults(fn=cmd_judge)
    p = sub.add_parser("score")
    p.add_argument("--split", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--others", nargs="+", required=True)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("verify")
    p.add_argument("--run-id", required=True)
    p.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a.fn(a)


if __name__ == "__main__":
    main()
