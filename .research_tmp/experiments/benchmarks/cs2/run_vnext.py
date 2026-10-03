# -*- coding: utf-8 -*-
"""CS2 新管线跑批：answer_pipeline.answer() × KB v2，输出 judge 输入 + 完整轨迹。

环境：KNOWLEDGE_CUTOFF=2025-05（自动设置）；KB 只读（base_kb_v2，答题期不写库）。
用法：python run_vnext.py --split dev --offset 0 --limit 30 --tag v1 [--no-kb] [--no-ext] [--cite] [--workers 3]
产物：arm_vnext/answers_<tag>.json（含 trace）、judge_input_vnext_<tag>.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

os.environ.setdefault("KNOWLEDGE_CUTOFF", "2025-05")
# 实测 GPUStack 27B 8 路并发与单路同速（7.4s vs 7.6s 墙钟）——瓶颈在客户端闸门而非服务端。
# 写作调用 max_tokens 5000（<8000）走共享车道；把共享车道放宽到 12。
os.environ.setdefault("LOCAL_MAX_CONCURRENT", "48")  # 10-03 压测：96 路内吞吐仍升，48 路内延迟基本不涨
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_shared", "tools"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import answer_pipeline as AP  # noqa: E402

RUB = {"dev": "sqa2_rubrics_v1_recomputed.json", "test": "sqa2_rubrics_v2_recomputed.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-kb", action="store_true")
    ap.add_argument("--no-ext", action="store_true")
    # 10-03 用户裁定：拿不到 S2 key → 只用 Sciverse；Sciverse 无引用图接口（实测），引文扩展默认关闭
    ap.add_argument("--cite", action="store_true", help="开启 S2 引文扩展（需 S2 配额；默认关）")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    # 10-03 用户裁定（方案 A）：CS2 默认整篇 ~1,000 词。依据：同 15 题长度对照，限 1,000 词 0.842 vs 无上限 0.812
    # （+0.031 [+0.006,+0.050]；官方 simplified_eval 无长度项但 AP 逐段计离题），且与 harness（843 词）长度对齐后
    # 优势仍 +0.099 [+0.049,+0.149]、全部来自 CR/CP。无上限（--word-budget 0）作为消融报告。
    ap.add_argument("--word-budget", type=int, default=1000, help="整篇总词数上限（默认 1000；0=无上限，消融用）")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    qs = json.load(open(os.path.join(HERE, "..", "scholarqa_multi", RUB[a.split]), encoding="utf-8"))
    qs = qs[a.offset:a.offset + a.limit]
    out_dir = os.path.join(HERE, "arm_vnext")
    os.makedirs(out_dir, exist_ok=True)
    ans_p = os.path.join(out_dir, f"answers_{a.tag}.json")
    done = {r["qid"]: r for r in json.load(open(ans_p, encoding="utf-8"))} if os.path.exists(ans_p) else {}
    kb = None if a.no_kb else AP.KB(os.path.join(HERE, "base_kb_v2"))
    cfg = {"split": a.split, "offset": a.offset, "limit": a.limit, "kb": not a.no_kb, "ext": not a.no_ext,
           "cite": a.cite, "screen": not a.no_screen, "state": not a.no_state, "probe": not a.no_probe, "word_budget": a.word_budget or None, "model": AP.MODEL, "cutoff": os.environ.get("KNOWLEDGE_CUTOFF")}
    json.dump(cfg, open(os.path.join(out_dir, f"config_{a.tag}.json"), "w"), indent=1)
    print("[vnext]", cfg, f"todo {sum(1 for q in qs if q['case_id'][:24] not in done)}", flush=True)

    def one(q):
        qid = q["case_id"][:24]
        t = time.time()
        try:
            r = AP.answer(q["question"], kb, use_ext=not a.no_ext, use_cite=a.cite, use_screen=not a.no_screen, use_state=not a.no_state,
                          use_probe=not a.no_probe, word_budget=a.word_budget or None)
            r.update({"qid": qid, "question": q["question"]})
        except Exception as e:
            r = {"qid": qid, "question": q["question"], "sections": [], "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"  {qid} {time.time() - t:.0f}s words={(r.get('trace') or {}).get('words')} "
              f"ev={(r.get('trace') or {}).get('by_src')} err={r.get('error')}", flush=True)
        return r

    todo = [q for q in qs if q["case_id"][:24] not in done]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(one, todo):
            done[r["qid"]] = r
            json.dump(list(done.values()), open(ans_p, "w", encoding="utf-8"), ensure_ascii=False)
    rows = [{"qid": r["qid"], "question": r["question"], "sections": r.get("sections") or []} for r in done.values()]
    json.dump(rows, open(os.path.join(HERE, f"judge_input_vnext_{a.tag}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[vnext] wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
