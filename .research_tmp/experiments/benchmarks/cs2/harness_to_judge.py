# -*- coding: utf-8 -*-
"""harness 臂答案 → judge 输入（与 judge_dev20.harness_sections 同口径：取 Claude Code 结果里第一个 JSON 对象的 sections）。
解析失败/超时的题不丢：以空 sections 写入（判分时四项计 0，避免幸存者偏差，REBUILD-PLAN E7）。
用法：python harness_to_judge.py <answers_json> <out_judge_input_json> [--split dev --offset 10 --limit 15]"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RUB = {"dev": "sqa2_rubrics_v1_recomputed.json", "test": "sqa2_rubrics_v2_recomputed.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("out")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    qs = json.load(open(os.path.join(HERE, "..", "scholarqa_multi", RUB[a.split]), encoding="utf-8"))[a.offset:a.offset + a.limit]
    rows = {r["qid"]: r for r in json.load(open(a.answers, encoding="utf-8"))}
    out, n_ok = [], 0
    for q in qs:
        qid = q["case_id"][:24]
        r = rows.get(qid) or {}
        secs = []
        res = r.get("result") or ""
        i, j = res.find("{"), res.rfind("}") + 1
        try:
            obj = json.loads(res[i:j]) if i >= 0 and j > i else {}
            secs = obj.get("sections") or []
        except Exception:
            secs = []
        n_ok += bool(secs)
        out.append({"qid": qid, "question": q["question"], "sections": secs})
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(out)} rows ({n_ok} with sections, {len(out) - n_ok} empty -> scored 0)")


if __name__ == "__main__":
    main()
