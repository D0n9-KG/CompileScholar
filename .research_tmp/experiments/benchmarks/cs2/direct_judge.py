# -*- coding: utf-8 -*-
"""直接判分（绕过 inspect eval 框架）：官方 scorer 函数 + 我们的事件循环。

inspect eval 反复卡死（多轮重启无解），但 scorer 本身是纯 async 函数——
直接构造 TaskState 调用，跳过框架的调度层。判分语义与官方完全一致
（同一套 score_sqa/score_precision/score_citation 代码）。
"""
import asyncio
import json
import sys
import os
import httpx

# Paratera httpx fix（Connection: close）——同步+异步都要打：
# scorer 走 inspect 的 async provider（AsyncClient），只打同步 Client
# 时异步调用照样挂死（elicit 判分 1.5h 零产出的根因）
_orig_send = httpx.Client.send
def _send_close(self, request, **kwargs):
    request.headers["Connection"] = "close"
    return _orig_send(self, request, **kwargs)
httpx.Client.send = _send_close
_orig_asend = httpx.AsyncClient.send
async def _asend_close(self, request, **kwargs):
    request.headers["Connection"] = "close"
    return await _orig_asend(self, request, **kwargs)
httpx.AsyncClient.send = _asend_close

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\scratch\ai2_baseline_2026-09-08\asta\asta-bench-main")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target

ENVF = r"C:\Users\D0n9\Desktop\CompileScholar\.env"
for line in open(ENVF, encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()
os.environ["OPENAI_API_KEY"] = "sk-placeholder"
# provider 环境映射：openai-api/glm/GLM-5.3 需要 GLM_* 键（.env 里是 PARATERA_*）
os.environ["GLM_API_KEY"] = os.environ.get("PARATERA_API_KEY", "")
os.environ["GLM_BASE_URL"] = os.environ.get("PARATERA_BASE_URL", "")

from astabench.evals.sqa.task import score_sqa, score_precision, score_citation
from astabench.evals.sqa.rubric import extract_json_from_response

# 构造最小 TaskState（scorer 只用 state.output.completion 和 metadata）
class _Out:
    def __init__(self, completion):
        self.completion = completion
class _State:
    def __init__(self, completion, metadata):
        self.output = _Out(completion)
        self.metadata = metadata

async def judge_one(question, rubric_json, report_json_str):
    """rubric_json=官方 rubric 行（含 ingredients），report=答案 JSON 串。"""
    model = get_model("openai-api/glm/GLM-5.3")
    state = _State(report_json_str, {
        "initial_prompt": question,
    })
    target = Target([json.dumps(rubric_json, indent=2)])
    scores = {}
    # 参数对齐官方 task.py（simplified_eval+assess_jointly=True 是
    # score_all 的默认；citation 走 all_at_once——与官方一致）
    from astabench.evals.sqa.task import score_sqa, score_precision, score_citation
    scorers = [
        ("ingredient_recall", score_sqa(model, simplified_eval=True,
                                         assess_jointly=True,
                                         temperature=0.5, top_p=0.95)),
        ("answer_precision", score_precision(model)),
        ("citation", score_citation(model, sentence_wise_cit_eval=False,
                                     all_at_once=True,
                                     temperature=0.5, top_p=0.95)),
    ]
    for name, sc in scorers:
        s = await sc(state, target)
        v = s.value if isinstance(s.value, dict) else {"score": s.value}
        scores[name] = v
    return scores

async def main():
    # argv: [input_json, out_json]（缺省=elicit dev20，向后兼容）
    in_path = sys.argv[1] if len(sys.argv) > 1 else "judge_input_elicit_dev20.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "direct_scores_elicit.json"
    rows = json.load(open(in_path, encoding="utf-8"))
    rubrics = json.load(open(r"..\scholarqa_multi\sqa2_rubrics_v1_recomputed.json", encoding="utf-8"))
    rmap = {q["case_id"][:24]: q for q in rubrics}
    done = {}
    if os.path.exists(out_path):
        done = json.load(open(out_path, encoding="utf-8"))
    for i, r in enumerate(rows):
        if r["qid"] in done:
            continue
        rubric = rmap[r["qid"]]
        report = json.dumps({"sections": r["sections"]}, ensure_ascii=False)
        print(f"[{i+1}/{len(rows)}] {r['qid'][:14]} judging...", flush=True)
        try:
            scores = await judge_one(r["question"], rubric, report)
            done[r["qid"]] = scores
            json.dump(done, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            flat = {k: (v.get("score") if isinstance(v, dict) else v) for k, v in scores.items()}
            print(f"  -> {flat}", flush=True)
        except Exception as e:
            print(f"  -> FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)
            done[r["qid"]] = {"error": str(e)[:200]}
            json.dump(done, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

asyncio.run(main())
