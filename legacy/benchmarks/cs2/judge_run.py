# -*- coding: utf-8 -*-
"""Run the official scorer over judge_input_{arm}_dev20.json rows.

Per question: memorized_solver -> official sqa scorer (GLM-5.3).
Writes per-question .eval logs + a summary scores json.

Usage:
  PYTHONUTF8=1 python judge_run.py <arm>   # harness | elicit | ours
Env: GLM_API_KEY / GLM_BASE_URL / HF_TOKEN / OPENAI_API_KEY(placeholder)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

CS2 = Path(__file__).resolve().parent
JUDGE_DIR = CS2 / "judge_runs"
JUDGE_DIR.mkdir(exist_ok=True)

TASK_TMPL = '''# -*- coding: utf-8 -*-
from inspect_ai import Task, task
from inspect_ai.solver import solver
from astabench.evals.sqa.task import sqa

REPORT = {report_json!r}


@solver
def memorized_solver():
    async def solve(state, generate):
        state.output.completion = REPORT
        return state
    return solve


@task
def sqa_arm() -> Task:
    t = sqa(scorer_model="openai-api/glm/GLM-5.3", limit=1,
            with_search_tools=False)
    t.solver = memorized_solver()
    return t
'''


def build_rubric_task(rubric_row, report_sections):
    """官方 sqa(limit=1) 取 rubrics[0]——构造首行=目标题的临时 rubrics
    文件，通过 astabench 的 HF 缓存重定向不可行；改走另一条路：
    直接用 inspect 的 dataset 参数不行（sqa 内部加载）。
    务实方案：单题 task 文件 + rubrics 首行 hack —— astabench 从
    hf_hub_download 拉 pinned revision 的 rubrics；我们本地已有
    rubrics 全量文件，临时改写 datasets 缓存成本高。
    → 最终方案：fork 一份 task 调用，dataset 直接传 MemoryDataset
    （复用官方 json_to_sample），scorer 原样。见 build_task_src。
    """
    return None


def build_task_src(question, sections):
    return TASK_TMPL.format(
        report_json=json.dumps({"sections": sections},
                               ensure_ascii=False))


def main():
    arm = sys.argv[1]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = json.load(open(CS2 / f"judge_input_{arm}_dev20.json",
                          encoding="utf-8"))
    # 裁到 dev20 集合（rubrics 前 20 的 qid）
    rubrics = json.load(open(
        CS2.parent / "scholarqa_multi" /
        "sqa2_rubrics_v1_recomputed.json", encoding="utf-8"))[:20]
    qids = {q["case_id"][:24] for q in rubrics}
    rows = [r for r in rows if r["qid"] in qids][:20]

    ENVF = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.env")
    env = {**os.environ.copy()}
    for line in open(ENVF, encoding="utf-8"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    env.update({
        "PYTHONUTF8": "1",
        "OPENAI_API_KEY": "sk-placeholder-not-used",
        "GLM_API_KEY": env.get("PARATERA_API_KEY", ""),
        "GLM_BASE_URL": env.get("PARATERA_BASE_URL", ""),
        "HF_TOKEN": env.get("HUGGING_FACE_TOKEN", ""),
    })

    scores = []
    out_scores = JUDGE_DIR / f"scores_{arm}_dev20.json"
    done_qids = set()
    if out_scores.exists():
        scores = json.load(open(out_scores, encoding="utf-8"))
        done_qids = {s["qid"] for s in scores}

    for i, r in enumerate(rows):
        if r["qid"] in done_qids:
            continue
        # 每题一个 task 文件（REPORT 内嵌该题 sections）
        task_file = JUDGE_DIR / f"task_{arm}_{r['qid']}.py"
        task_file.write_text(
            build_task_src(r["question"], r["sections"]),
            encoding="utf-8")
        log_dir = JUDGE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        print(f"[{arm}] judging {i+1}/{len(rows)}: {r['qid']}",
              flush=True)
        # 注意：官方 sqa(limit=1) 判的是 rubrics[0]——不是本题！
        # 需要让题对齐：临时 rubrics 重排不可行，改为检查本题是否
        # rubrics[0]；非则跳过并记录（单独跑的题位偏移问题——
        # 用 CS2_RUBRIC_FIRST 环境变量+astabench monkeypatch 重定向）
        # 简化：跑之前确认 rubrics[0].case_id 前缀 == r.qid，否则改
        # 用多题连续跑（把 20 题全部喂进一个 task：memorized solver
        # 按 sample 顺序吐对应答案）——这才是正解，见下
        break

    print("[hint] 单题对齐问题——改用整批单 task 方案（下一步实现）")


if __name__ == "__main__":
    main()
