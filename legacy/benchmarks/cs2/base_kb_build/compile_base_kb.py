# -*- coding: utf-8 -*-
"""Compile the CS2 base KB: registry + views over the merged records.

Inputs (all in base_kb/):
  coarse_records.json      Tier-1 (7,824 coarse records)
  survey_records_adapted.json   survey S1-S4 adapted (lineage/absence
                                shapes + consensus layer)
  hub records (records_hub.json) — optional, when hub deep pass done
Output:
  registry_cs2.json  (via registry_growth: survey/coarse entity
                      surfaces -> entities)
  dim_vocab_cs2.json (light vocab from coarse+survey subjects)
  views_cs2.json     (views compiler over merged records)
"""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
            r"\experiments\benchmarks\cs2\base_kb")
BUILD = BASE.parent / "base_kb_build"
SRC = Path(r"C:\Users\D0n9\Desktop\CompileScholar\src")


def merge_records() -> str:
    """coarse + survey-adapted (+ hub) -> records_merged.json"""
    merged = {}
    coarse = json.load(open(BASE / "coarse_records.json",
                            encoding="utf-8"))
    for k, v in coarse.items():
        if isinstance(v, dict) and v.get("records"):
            merged[k] = v
    survey = json.load(open(BASE / "survey_records_adapted.json",
                            encoding="utf-8"))
    merged.update(survey)
    hub_p = BASE / "records_hub.json"
    if hub_p.exists():
        hub = json.load(open(hub_p, encoding="utf-8"))
        merged.update(hub)
    out = BASE / "records_merged.json"
    json.dump(merged, open(out, "w", encoding="utf-8"),
              ensure_ascii=False)
    n = sum(len(v.get("records", [])) for v in merged.values())
    print(f"[merge] {len(merged)} papers, {n} records -> {out.name}")
    return str(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rec_path = merge_records()

    # 1) registry growth：从 merged records 长实体注册表
    #    registry_growth 吃 records_slot 形态，输出 registry v2
    env = {"PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8",
           "LOCAL_MAX_CONCURRENT": "8"}
    import os
    env = {**os.environ.copy(), **env}
    r = subprocess.run(
        [sys.executable, "-m", "kb_compiler.records.registry_growth",
         "--registry", str(BASE / "_seed_registry.json"),
         "--records", rec_path,
         "--out-dir", str(BASE), "--model", "local:Qwen3.8-27B"],
        cwd=str(SRC), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    if r.returncode != 0:
        print("registry_growth FAILED:", r.stderr[-1500:], flush=True)
        sys.exit(1)

    # 2) vocab：从 coarse subjects 轻构建（registry_growth 产物 +
    #    subject 家族归纳走 registry.py 主链——轻路线：直接用
    #    registry_growth 的实体清单做 vocab 种子）
    # 3) views：views 编译器
    r2 = subprocess.run(
        [sys.executable, "-m", "kb_compiler.views.compiler",
         "--records", rec_path,
         "--registry", str(BASE / "registry_growth" / "registry_v2.json"),
         "--vocab", str(BASE / "dim_vocab_cs2.json"),
         "--manifest", str(BASE / "manifest_all.json"),
         "--out", str(BASE / "views_cs2.json")],
        cwd=str(SRC), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    print(r2.stdout[-2000:] if r2.stdout else "", flush=True)
    if r2.returncode != 0:
        print("views FAILED:", r2.stderr[-1500:], flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
