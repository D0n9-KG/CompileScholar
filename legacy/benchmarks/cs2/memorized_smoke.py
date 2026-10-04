# -*- coding: utf-8 -*-
"""Memorized-arm judge smoke: Elicit x official scorer x GLM-5.3 (1 sample).

Validates the re-judging pipeline end-to-end: Elicit sections JSON ->
official sqa scorer (memorized_solver mode) -> four-facet scores.
"""
import json
import sys
from pathlib import Path

CS2 = Path(__file__).resolve().parent
sys.path.insert(0, str(CS2))

# elicit dev rows -> memorized solver 报告（取已匹配 qid 的第一条）
rows = json.load(open(CS2 / "arm_memorized" / "answers_elicit_dev.json",
                      encoding="utf-8"))
row = rows[0]
REPORT = json.dumps({"sections": row["sections"]}, ensure_ascii=False)

# 找到对应 rubric（按 case_id）并注入 smoke task
rubrics = json.load(open(
    CS2.parent / "scholarqa_multi" / "sqa2_rubrics_v1_recomputed.json",
    encoding="utf-8"))
target = next(q for q in rubrics if q["case_id"][:24] == row["qid"])
print(f"[smoke] question: {target['question'][:80]}")
print(f"[smoke] elicit sections: {len(row['sections'])} | "
      f"sample title: {row['sections'][0].get('title')}")

# 生成单题 smoke task 文件（复用 0.4 的 memorized_solver 模式）
task_src = (CS2 / "judge_smoke_task.py")
smoke = f'''# -*- coding: utf-8 -*-
from inspect_ai import Task, task
from inspect_ai.solver import solver
from astabench.evals.sqa.task import sqa

REPORT = {REPORT!r}


@solver
def memorized_solver():
    async def solve(state, generate):
        state.output.completion = REPORT
        return state
    return solve


@task
def sqa_elicit_glm() -> Task:
    t = sqa(scorer_model="openai-api/glm/GLM-5.3", limit=1,
            with_search_tools=False)
    t.solver = memorized_solver()
    return t
'''
# 注意：官方 sqa(limit=1) 取 rubrics[0]——我们的目标题必须在第 0 位。
# 直接构造 reorder 版 rubrics 太重；换法：找到目标题在 rubrics 的 idx，
# 若非 0 则提示手动跑该 idx 的方法（smoke 只验证管线，用 idx=0 的
# Elicit 匹配即可——重新选 row 使 qid == rubrics[0].case_id[:24]）
first_qid = rubrics[0]["case_id"][:24]
match = next((r for r in rows if r["qid"] == first_qid), None)
if match is None:
    print(f"[smoke] NOTE: elicit 无 rubrics[0] 的匹配——"
          f"用有限样本验证（idx 偏移）")
    match = row

smoke = smoke.replace(
    "REPORT = {REPORT!r}",
    f"REPORT = {json.dumps({'sections': match['sections']}, ensure_ascii=False)!r}")
open(CS2 / "judge_smoke_task.py", "w", encoding="utf-8").write(smoke)
print("[smoke] task written: judge_smoke_task.py")
