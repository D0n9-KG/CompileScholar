# -*- coding: utf-8 -*-
"""Generate judge_batch_task.py with an arm's answers inlined.

Usage: python gen_judge_task.py <arm>   # harness | elicit | ours
"""
import json
import sys
from pathlib import Path

CS2 = Path(__file__).resolve().parent

TMPL = '''# -*- coding: utf-8 -*-
"""Auto-generated: {arm} dev20 batch judge task (do not edit by hand)."""
from inspect_ai import Task, task
from inspect_ai.solver import solver
from astabench.evals.sqa.task import sqa

ANSWERS = {answers_lit}


@solver
def memorized_solver():
    async def solve(state, generate):
        q = (state.metadata or {{}}).get("initial_prompt")
        state.output.completion = ANSWERS.get(q, "")
        return state
    return solve


@task
def sqa_{arm}() -> Task:
    t = sqa(scorer_model="openai-api/glm/GLM-5.3", limit=20,
            with_search_tools=False)
    t.solver = memorized_solver()
    return t
'''


def main():
    arm = sys.argv[1]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = json.load(open(CS2 / f"judge_input_{arm}_dev20.json",
                          encoding="utf-8"))
    answers = {r["question"]: json.dumps({"sections": r["sections"]},
                                         ensure_ascii=False)
               for r in rows}
    src = TMPL.format(arm=arm,
                      answers_lit=json.dumps(answers,
                                             ensure_ascii=False, indent=1))
    out = CS2 / "judge_batch_task.py"
    out.write_text(src, encoding="utf-8")
    print(f"[gen] {arm}: {len(answers)} answers inlined -> {out.name}")
    print(f"[run] PYTHONUTF8=1 OPENAI_API_KEY=x python -m inspect_ai "
          f"eval judge_batch_task.py@sqa_{arm}")


if __name__ == "__main__":
    main()
