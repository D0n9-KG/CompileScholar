# -*- coding: utf-8 -*-
"""C⑥ gate: 50 篇新旧覆盖率对照——旧深抽的信息要被新记录覆盖（INTEGRATED-SYSTEM-1005 §12 阶段 C 闸门）。

旧侧：data/benchmarks/cs2/kb_v2/records.json（旧记录层，1,262 篇，arxiv_<id> 键）。
新侧：data/derived/extract.sqlite 的 statements（speaker = arxiv:<id>）。
取两侧都有的论文（不足 50 篇时全取并如实报告；缺全文的论文可先用 cli acquire/parse/build 补齐再跑）。
每篇把旧记录分批交给本地模型，对着该篇全部新表述判"信息是否被覆盖"（yes / partial / no + 指向哪条新记录）。
产出 runs/gate_c/coverage.json：每篇覆盖率、总体 yes+partial/yes 两种口径、未覆盖样本。
Run: python experiments/extract/gate_coverage.py [--n 50] [--workers 8]"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from collections import defaultdict

from compilescholar.core import config as C, paths
from compilescholar.dfc.store import parallel
from compilescholar.llm import client as LC

JUDGE = """One research paper was extracted twice: by an OLD system (records below) and by a NEW system
(statements below). For EVERY old record, decide whether the NEW statements carry the same information
(a statement counts when its content matches, even with different wording; numbers must match exactly).

Old records (JSON list, each with an index "i"):
{old}

New statements (numbered list "pass | facet | role | text | value/condition"):
{new}

Answer with JSON only:
{{"covered": [{{"i": <old index>, "verdict": "yes" | "partial" | "no", "by": [<new statement numbers>]}}]}}"""


def _new_lines(rows) -> str:
    out = []
    for j, (pass_name, facet, role, text, meta) in enumerate(rows, 1):
        try:
            m = json.loads(meta)
        except (TypeError, ValueError):
            m = {}
        extra = m.get("value") or (m.get("config") or {}).get("value") or ""
        out.append(f"[{j}] {pass_name} | {facet} | {role} | {text[:200]}"
                   + (f" | value={extra}" if extra else ""))
    return "\n".join(out)


def main(n: int, workers: int, config: str | None) -> None:
    cfg = C.load(config, [])
    LC.configure(json.loads(json.dumps(cfg.llm or {})), run_id=f"gate-coverage-{time.strftime('%Y%m%dT%H%M%S')}",
                 caller="gate")
    old = json.load(open("data/benchmarks/cs2/kb_v2/records.json", encoding="utf-8"))
    old_by_pid = {}
    for k, v in old.items():
        if k.startswith("arxiv_"):
            pid = "arxiv:" + k[len("arxiv_"):]
            recs = (v or {}).get("records") or []
            if recs:
                old_by_pid[pid] = recs
    con = sqlite3.connect(f"file:{(paths.derived() / 'extract.sqlite').as_posix()}?mode=ro", uri=True)
    new_by_pid = defaultdict(list)
    for row in con.execute("SELECT speaker, pass, facet, role, text, meta FROM statements"):
        new_by_pid[row[0]].append(row[1:])
    con.close()
    both = sorted(set(old_by_pid) & {p for p in new_by_pid if new_by_pid[p]})
    print(f"[gate] old-deep papers {len(old_by_pid)}, new-extracted {len(new_by_pid)}, intersection {len(both)}")
    papers = random.Random(20261007).sample(both, min(n, len(both)))

    jobs = []                       # (pid, old-chunk)
    for pid in papers:
        recs = old_by_pid[pid]
        slim = [{"i": i, "kind": r.get("kind"), "claim": (r.get("claim") or r.get("text") or "")[:220],
                 "quote": (r.get("quote") or "")[:220],
                 "value": ((r.get("measure") or {}).get("value") if isinstance(r.get("measure"), dict) else None)}
                for i, r in enumerate(recs)]
        for i in range(0, len(slim), 20):
            jobs.append((pid, slim[i:i + 20]))
    print(f"[gate] {len(papers)} papers, {sum(len(old_by_pid[p]) for p in papers)} old records, "
          f"{len(jobs)} judge calls")

    results: dict = defaultdict(list)

    def one(job):
        pid, chunk = job
        prompt = JUDGE.format(old=json.dumps(chunk, ensure_ascii=False), new=_new_lines(new_by_pid[pid])[:12000])
        raw = LC.call_local(prompt, max_tokens=2500, temperature=0.0, enable_thinking=False, item=pid)
        try:
            obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            got = {c["i"]: c for c in obj.get("covered") or [] if isinstance(c, dict)}
        except Exception:
            got = {}
        out = []
        for c in chunk:
            g = got.get(c["i"]) or {}
            out.append({"i": c["i"], "kind": c["kind"], "claim": c["claim"][:120],
                        "verdict": g.get("verdict") or "unjudged", "by": g.get("by") or []})
        results[pid] += out

    parallel(one, jobs, workers, every=50, label="gate:coverage")
    per_paper = {}
    for pid, recs in results.items():
        yes = sum(1 for r in recs if r["verdict"] == "yes")
        part = sum(1 for r in recs if r["verdict"] == "partial")
        per_paper[pid] = {"old_records": len(recs), "yes": yes, "partial": part,
                          "coverage_strict": round(yes / max(1, len(recs)), 3),
                          "coverage_lenient": round((yes + part) / max(1, len(recs)), 3)}
    tot = sum(v["old_records"] for v in per_paper.values())
    yes = sum(v["yes"] for v in per_paper.values())
    part = sum(v["partial"] for v in per_paper.values())
    uncovered = [r for pid, recs in results.items() for r in recs if r["verdict"] == "no"][:60]
    dest = paths.runs() / "gate_c"
    dest.mkdir(parents=True, exist_ok=True)
    summary = {"papers": len(per_paper), "old_records": tot, "yes": yes, "partial": part,
               "coverage_strict": round(yes / max(1, tot), 4),
               "coverage_lenient": round((yes + part) / max(1, tot), 4),
               "per_paper": per_paper, "uncovered_samples": uncovered}
    (dest / "coverage.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_paper", "uncovered_samples")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    main(a.n, a.workers, a.config)
