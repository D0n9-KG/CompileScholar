# -*- coding: utf-8 -*-
"""DeepScholar-Bench × 新答题管线（开放检索：KB v2 + 外部 + 引文扩展；按每篇目标论文发表日做知识截止）。

任务=给目标论文摘要写 Related Works。题面构造沿用 DSB 官方 query 模板（p6/oracle_inputs.json 的 query 字段）。
截止：每题 KNOWLEDGE_CUTOFF=目标论文 published_date 的年-月（检索层严格早于该月；只有年份时同年保守排除）。
另外从结果中剔除目标论文本身（标题相同）。
48 个唯一题（66 个 gt 目录去重，与 p6 测试床同口径；按每个 qid 最小目录号落盘，供 judge_nuggets.py / 官方 eval 使用）。
输出：p6 测试床格式 gen/<SYS>/<gt_dir>.md（正文，引用为 [k]，不附参考文献列表）+ 轨迹 json。
用法：python run_vnext_dsb.py --sys vnext_v1 [--limit N] [--workers 2] [--no-kb]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LOCAL_MAX_CONCURRENT", "48")  # 10-03 压测：96 路内吞吐仍升，48 路内延迟基本不涨  # 见 run_vnext.py：服务端 8 路并发不降速
sys.path.insert(0, os.path.join(HERE, "..", "_shared", "tools"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import answer_pipeline as AP  # noqa: E402

P6 = os.path.normpath(os.path.join(HERE, "..", "..", "..", "review_1002", "p6"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-kb", action="store_true")
    # 10-03 用户裁定：拿不到 S2 key → 只用 Sciverse；Sciverse 无引用图接口（实测），引文扩展默认关闭
    ap.add_argument("--cite", action="store_true", help="开启 S2 引文扩展（需 S2 配额；默认关）")
    a = ap.parse_args()
    rows = json.load(open(os.path.join(P6, "oracle_inputs.json"), encoding="utf-8"))
    if a.limit:
        rows = rows[:a.limit]
    gen = os.path.join(P6, "gen", a.sys)
    os.makedirs(gen, exist_ok=True)
    tr_dir = os.path.join(HERE, "arm_vnext_dsb", a.sys)
    os.makedirs(tr_dir, exist_ok=True)
    kb = None if a.no_kb else AP.KB(os.path.join(HERE, "base_kb_v2"))

    def one(r):
        out_p = os.path.join(gen, f"{r['gt_dir']}.md")
        if os.path.exists(out_p):
            return r["gt_dir"], "skip"
        cut = (r.get("published_date") or "")[:7]
        # 每题截止=目标论文发表年月；线程局部（cutoff.set_thread_cutoff），并发题互不干扰。
        # 注意 answer_pipeline 内部的检索线程池是新线程，需要在其中重设——由 AP.answer(cutoff=...) 传入。
        # DSB 任务给定目标论文摘要（官方 query 模板即含摘要）——作为任务材料证据（本文自述，不算外部引用）
        res = AP.answer(r["query"], kb, use_ext=True, use_cite=a.cite, cutoff=cut,
                        task_context={"title": r.get("title"), "text": r.get("abstract") or ""})
        tnorm = re.sub(r"\W+", " ", (r.get("title") or "").lower()).strip()
        body = []
        for s in res["sections"]:
            body.append(f"## {s['title']}\n\n{s['text']}")
        text = "## Related Works\n\n" + "\n\n".join(body)
        self_cited = [c["title"] for s in res["sections"] for c in s["citations"]
                      if re.sub(r"\W+", " ", (c["title"] or "").lower()).strip() == tnorm]
        open(out_p, "w", encoding="utf-8").write(text)
        json.dump({"gt_dir": r["gt_dir"], "qid": r["qid"], "cutoff": cut, "self_cited": self_cited,
                   "sections": res["sections"], "trace": {k: res["trace"][k] for k in
                   ("plan", "by_src", "assemble", "words", "elapsed_s")}},
                  open(os.path.join(tr_dir, f"{r['gt_dir']}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        return r["gt_dir"], f"ok words={res['trace']['words']} ev={res['trace']['by_src']} self={len(self_cited)}"

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for f in cf.as_completed([ex.submit(one, r) for r in rows]):
            try:
                print(*f.result(), flush=True)
            except Exception as e:
                print("ERR", str(e)[:200], flush=True)


if __name__ == "__main__":
    main()
