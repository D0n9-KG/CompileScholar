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

# ── 提速补丁（EXPERIMENT-DESIGN-REV-0928.md 方案 A+C）───────────────
# A: 官方 generate_with_retry 默认 max_retries=20（21 次尝试），GLM 思考
#    模型 JSON 畸形时一次能烧 20+ 次重试。封顶 4（5 次尝试）；判分语义
#    不变（重试只影响失败容忍度，配合"全零题重判一次"排雷规则）。
import astabench.evals.sqa.retry_utils as _ru
import astabench.evals.sqa.rubric as _mod_rubric
import astabench.evals.sqa.precision_eval as _mod_precision
import astabench.evals.sqa.citation_eval as _mod_citation
_orig_gwr = _ru.generate_with_retry
async def _gwr_capped(*a, **kw):
    kw.setdefault("max_retries", 4)
    return await _orig_gwr(*a, **kw)
for _m in (_mod_rubric, _mod_precision, _mod_citation):
    _m.generate_with_retry = _gwr_capped

# 构造最小 TaskState（scorer 只用 state.output.completion 和 metadata）
class _Out:
    def __init__(self, completion):
        self.completion = completion
class _State:
    def __init__(self, completion, metadata):
        self.output = _Out(completion)
        self.metadata = metadata

# C: 全局 scorer 并发闸——题间并行 × 每题 3 scorer 并行后，最坏 8×3=24
# 路在飞；用全局信号量压到 CAP，防 Paratera 限流风暴（题内并行后题级
# 信号量不再控制实际调用量）。
_SCORER_CAP = 12
_scorer_sem = None  # main() 里初始化（绑定事件循环）

async def judge_one(question, rubric_json, report_json_str):
    """rubric_json=官方 rubric 行（含 ingredients），report=答案 JSON 串。"""
    model = get_model("openai-api/glm/GLM-5.3")
    target = Target([json.dumps(rubric_json, indent=2)])
    # 参数对齐官方 task.py（simplified_eval+assess_jointly=True 是
    # score_all 的默认；citation 走 all_at_once——与官方一致）
    scorers = [
        ("ingredient_recall", score_sqa(model, simplified_eval=True,
                                         assess_jointly=True,
                                         temperature=0.5, top_p=0.95)),
        ("answer_precision", score_precision(model)),
        ("citation", score_citation(model, sentence_wise_cit_eval=False,
                                     all_at_once=True,
                                     temperature=0.5, top_p=0.95)),
    ]
    # C: 三 scorer 并行（每个拿独立 _State，防读共享实例的竞态）
    # FULLCHAIN-AUDIT P2-15：保存 scorer metadata（per-criterion 明细/
    # num_retries/irrelevant_texts）——排雷与审稿举证可审计；P2-16：
    # return_exceptions=True（一 scorer 崩不再丢弃其余两个的结果）
    async def _run(name, sc):
        async with _scorer_sem:
            state = _State(report_json_str, {"initial_prompt": question})
            s = await sc(state, target)
        v = s.value if isinstance(s.value, dict) else {"score": s.value}
        md = getattr(s, "explanation", None)
        if md:
            try:
                v["_meta"] = json.loads(md) if isinstance(md, str) else md
            except Exception:
                v["_meta_explanation"] = str(md)[:2000]
        return name, v
    pairs = await asyncio.gather(*[_run(n, sc) for n, sc in scorers],
                                 return_exceptions=True)
    out = {}
    for p in pairs:
        if isinstance(p, Exception):
            # 单 scorer 崩：记录到 _errors 键（resume 可重试），不丢其他
            out.setdefault("_errors", []).append(
                f"{type(p).__name__}: {str(p)[:150]}")
        else:
            out[p[0]] = p[1]
    return out

async def main():
    # argv: [input_json, out_json, parallel]（缺省=elicit dev20；parallel
    # =题间并发路数，缺省 8——官方 inspect eval 本就 --parallel 题级并行，
    # 题间完全独立；每题内 3 scorer 也已并行（方案 C），全局并发由
    # _scorer_sem 压到 _SCORER_CAP 防限流风暴）
    in_path = sys.argv[1] if len(sys.argv) > 1 else "judge_input_elicit_dev20.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "direct_scores_elicit.json"
    n_par = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    global _scorer_sem
    _scorer_sem = asyncio.Semaphore(_SCORER_CAP)
    rows = json.load(open(in_path, encoding="utf-8"))
    rubrics = json.load(open(r"..\scholarqa_multi\sqa2_rubrics_v1_recomputed.json", encoding="utf-8"))
    rmap = {q["case_id"][:24]: q for q in rubrics}
    done = {}
    if os.path.exists(out_path):
        done = json.load(open(out_path, encoding="utf-8"))
    todo = [r for r in rows if r["qid"] not in done]
    # FULLCHAIN-AUDIT P2-16：带 _errors 或 error 键的题重试一次（之前
    # error 落进 done 后 resume 永久跳过）
    retry = [qid for qid, v in done.items()
             if isinstance(v, dict) and ("error" in v or "_errors" in v)]
    if retry:
        for qid in retry:
            done.pop(qid)
        todo = [r for r in rows if r["qid"] not in done]
        print(f"[judge] retrying {len(retry)} error rows", flush=True)
    print(f"[judge] {len(todo)} to judge ({len(done)} resumed) | "
          f"parallel={n_par}", flush=True)

    import asyncio as _aio
    sem = _aio.Semaphore(n_par)
    save_lock = _aio.Lock()

    async def _one(i, r):
        rubric = rmap[r["qid"]]
        report = json.dumps({"sections": r["sections"]}, ensure_ascii=False)
        async with sem:
            print(f"[{i+1}/{len(rows)}] {r['qid'][:14]} judging...", flush=True)
            try:
                scores = await judge_one(r["question"], rubric, report)
            except Exception as e:
                print(f"  -> FAIL {r['qid'][:14]} {type(e).__name__}: "
                      f"{str(e)[:100]}", flush=True)
                scores = {"error": str(e)[:200]}
            async with save_lock:
                done[r["qid"]] = scores
                json.dump(done, open(out_path, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            flat = {}
            for k, v in scores.items():
                if isinstance(v, dict):
                    nums = {kk: vv for kk, vv in v.items()
                            if isinstance(vv, (int, float))}
                    flat[k] = (list(nums.values())[0] if nums else None)
                else:
                    flat[k] = v
            print(f"  -> {r['qid'][:14]} {flat}", flush=True)

    await _aio.gather(*[_one(rows.index(r), r) for r in todo])

asyncio.run(main())
