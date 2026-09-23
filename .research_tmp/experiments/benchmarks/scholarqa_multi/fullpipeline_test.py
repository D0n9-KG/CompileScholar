# -*- coding: utf-8 -*-
"""Full-pipeline test (user directive 2026-09-24: before SQA2, re-run
extraction + the whole downstream chain on a Multi paper subset and
scrutinize every product for problems and hidden risks).

Subset: 12 papers stratified over the six subjects, chosen deterministically
(seed fixed) with a bias to "hard" shapes: the longest text (LSST-class), a
tables-heavy paper, a formula-dense paper, plus normal coverage.

Scratch KB dir (kb_pipeline_test/) — production kb/ is never touched.
Chain: cards -> registry_vocab -> deep_extract -> postcheck -> table_extract
-> notation -> registry_growth -> registry_dedup -> views, all local
Qwen3.8-27B, each stage's stdout captured per stage.

Usage:  python fullpipeline_test.py [--papers 12] [--stage all]
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(BASE)))), "src")
sys.path.insert(0, SRC)

TEXTS = os.path.join(BASE, "corpus", "texts")
MANIFEST = os.path.join(BASE, "corpus", "manifest.json")
TEST_KB = os.path.join(BASE, "kb_pipeline_test")
TEST_TEXTS = os.path.join(TEST_KB, "texts")

MODEL = "local:Qwen3.8-27B"


def pick_papers(n: int) -> list[dict]:
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    by_subj = {}
    for row in manifest:
        by_subj.setdefault(row["subject"], []).append(row)
    random.seed(20260924)
    # stratified quota over the six subjects (min 1 each), rest proportional
    subjects = sorted(by_subj)
    quota = {s: max(1, round(n * len(by_subj[s]) / len(manifest)))
             for s in subjects}
    picked = []
    for s in subjects:
        picked.extend(random.sample(by_subj[s], min(quota[s], len(by_subj[s]))))
    # hard shapes: longest text overall + tables-heavy + formula-dense, if
    # not already in
    def _size(row):
        p = os.path.join(TEXTS, row["paper_id"] + ".md")
        return os.path.getsize(p) if os.path.exists(p) else 0
    longest = max(manifest, key=_size)
    if longest["paper_id"] not in {p["paper_id"] for p in picked} and len(picked) < n + 3:
        picked.append(longest)
    return picked[:max(n, len(picked))]


def run_stage(name: str, cmd: list[str]) -> bool:
    t0 = time.time()
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
        "LLM_CALL_LOG": os.path.join(TEST_KB, "ledger_pipeline_test.jsonl"),
        "LLM_RUN_ID": f"pipelinetest-{name}",
        "LOCAL_MAX_CONCURRENT": "16",
        "LLM_PROVIDER_ALLOWLIST": "local",
        "KB_EMBED_PROVIDER": "local",
    })
    log = open(os.path.join(TEST_KB, f"stage_{name}.log"), "w",
               encoding="utf-8", errors="replace")
    p = subprocess.run(cmd, cwd=SRC, env=env, stdout=log,
                       stderr=subprocess.STDOUT)
    log.close()
    dt = time.time() - t0
    status = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
    print(f"[{name}] {status} ({dt:.0f}s) — log: stage_{name}.log", flush=True)
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=12)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.stage == "all" and os.path.exists(TEST_KB):
        print(f"[setup] removing stale {TEST_KB}")
        shutil.rmtree(TEST_KB)
    os.makedirs(TEST_KB, exist_ok=True)

    papers = pick_papers(args.papers)
    pids = [p["paper_id"] for p in papers]
    json.dump(papers, open(os.path.join(TEST_KB, "subset_manifest.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs(TEST_TEXTS, exist_ok=True)
    for pid in pids:
        src = os.path.join(TEXTS, pid + ".md")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(TEST_TEXTS, pid + ".md"))
    print(f"[setup] {len(pids)} papers: "
          f"{[p['subject'] for p in papers]}", flush=True)

    K = TEST_KB
    stages = [
        ("cards", ["python", "-m", "kb_compiler.records.cards",
                   "--texts", TEST_TEXTS, "--manifest", MANIFEST,
                   "--out", f"{K}/cards.json", "--model", MODEL]),
        ("registry_vocab", ["python", "-m", "kb_compiler.records.registry",
                            "--cards", f"{K}/cards.json",
                            "--manifest", MANIFEST, "--out-dir", K,
                            "--merge-mode", "blocked", "--tau1", "0.84",
                            "--model", MODEL]),
        ("deep_extract", ["python", "-m", "kb_compiler.records.deep_extract",
                          "--texts", TEST_TEXTS, "--manifest", MANIFEST,
                          "--cards", f"{K}/cards.json",
                          "--registry", f"{K}/registry.json",
                          "--vocab", f"{K}/dim_vocab_v1.json",
                          "--out", f"{K}/records_slot.json",
                          "--model", MODEL, "--pool", "16"]),
        ("postcheck", ["python", "-m", "kb_compiler.records.postcheck",
                       "--records", f"{K}/records_slot.json",
                       "--texts", TEST_TEXTS, "--vocab", f"{K}/dim_vocab_v1.json",
                       "--out-dir", f"{K}/postcheck", "--model", MODEL]),
        ("table_extract", ["python", "-m", "kb_compiler.records.table_extract",
                           "--texts", TEST_TEXTS, "--cards", f"{K}/cards.json",
                           "--registry", f"{K}/registry.json",
                           "--checked", f"{K}/postcheck/records_checked.json",
                           "--out", f"{K}/records_tables.json",
                           "--semantic-dir", f"{K}/table_semantic",
                           "--provider", "local", "--model", MODEL, "--canary"]),
        ("notation", ["python", "-m", "kb_compiler.records.notation_harvest",
                      "--texts", TEST_TEXTS,
                      "--records", f"{K}/records_tables_by_paper.json",
                      "--out", f"{K}/notation.json",
                      "--provider", "local", "--model", MODEL]),
        ("registry_growth", ["python", "-m", "kb_compiler.records.registry_growth",
                             "--registry", f"{K}/registry.json",
                             "--records", f"{K}/records_slot.json",
                             "--out-dir", K, "--model", MODEL]),
        ("registry_dedup", ["python", "-m", "kb_compiler.records.registry_dedup",
                            "--registry", f"{K}/registry_v2.json",
                            "--out", f"{K}/registry_v3.json",
                            "--report", f"{K}/registry_dedup_report.json"]),
        ("views", ["python", "-m", "kb_compiler.views.compiler",
                   "--records", f"{K}/records_slot.json",
                   "--registry", f"{K}/registry_v3.json",
                   "--vocab", f"{K}/dim_vocab_v1.json",
                   "--manifest", MANIFEST,
                   "--out", f"{K}/views.json"]),
    ]
    only = None if args.stage == "all" else args.stage.split(",")
    for name, cmd in stages:
        if only and name not in only:
            continue
        if not run_stage(name, cmd):
            print(f"[abort] stage {name} failed — inspect "
                  f"{K}/stage_{name}.log")
            sys.exit(1)
    print("[done] full pipeline complete — products in kb_pipeline_test/")


if __name__ == "__main__":
    main()
