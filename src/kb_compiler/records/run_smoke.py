# -*- coding: utf-8 -*-
"""Smoke driver: pre-registered 5-paper x 2-arm end-to-end + canary + judge.

Pre-registration: .research_tmp/experiments/stageB/SMOKE-PREREG.md (v1, frozen
2026-09-05). Gates are evaluated EXACTLY as pre-registered; no softening.

Stages:
  extract : per arm -> canary card, slot pass (5 papers + canary), postcheck
            (live repair), canary scoring, timing, single-arm gates
  judge   : Kimi-K2.6 blind spot-check (10 records/arm, stratified) + A/B gate
  all     : extract then judge

Usage:
  python -m kb_compiler.records.run_smoke --stage all \
      --base ../.research_tmp/experiments/stageB \
      --texts ../.research_tmp/experiments/e2_need_gap/gold_work/texts
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .canary import CANARY_PID, CANARY_TEXT, score as canary_score
from .common import load_corpus, load_json, load_manifest, save_json
from .postcheck import run_postcheck
from .skeleton import build_card
from .slot import extract_paper

SMOKE_PAPERS = ["repro", "qr_dqn", "r2d2", "icm", "seed_PER"]
ARMS = {"dsf": "DeepSeek-V4-Flash", "qwen": "Qwen3.8-Max"}
CANARY_TITLE = "NOVA: Nested Overlap Value Aggregation for Sample-Efficient Control"

JUDGE_PROMPT = """你是抽取质量盲评裁判。给定一条从论文抽取的结构化记录和原文窗口，评三项（各1-5整数）：
1. quote忠实性：quote 是否逐字来自原文窗口，且真支持该记录
2. 字段绑定：数值/指标/方法名的对应是否正确（无错位、无张冠李戴）
3. 条件保留：记录是否保留了原文中的限定条件（协议/范围/条件词），未丢失或未添加

记录 JSON：
{record}

原文窗口：
{window}

输出 JSON：{{"quote":分,"binding":分,"condition":分,"note":"一句话"}}"""


def extract_arm(arm: str, model: str, base: str, texts_dir: str):
    t_arm = time.time()
    texts = dict(load_corpus(texts_dir))
    texts[CANARY_PID] = CANARY_TEXT
    manifest = load_manifest(f"{base}/manifest_rl40.json")
    cards = dict(load_json(f"{base}/paper_cards.json", {}))
    registry = load_json(f"{base}/registry.json", {})
    vocab = load_json(f"{base}/dim_vocab_v1.json", {})

    # canary card per arm (upstream cards shared per pre-reg)
    _, ccard = build_card(CANARY_PID, CANARY_TEXT, CANARY_TITLE, model)
    if ccard:
        cards[CANARY_PID] = ccard

    todo = SMOKE_PAPERS + [CANARY_PID]
    results, timing = {}, {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(extract_paper, pid, texts[pid], cards.get(pid) or {},
                          registry, vocab, model,
                          (manifest.get(pid) or {}).get("title") or CANARY_TITLE): pid
                for pid in todo}
        t0 = {pid: time.time() for pid in todo}
        for f in futs:
            pid, out = f.result()
            results[pid] = out
            timing[pid] = round(time.time() - t0[pid], 1)
    save_json(results, f"{base}/records_smoke_{arm}.json")

    checked, dropped, warnings, stats = run_postcheck(results, texts, vocab, model)
    save_json(checked, f"{base}/records_checked_{arm}.json")
    save_json(dropped, f"{base}/dropped_{arm}.json")
    save_json(warnings, f"{base}/warnings_{arm}.json")

    can = canary_score((checked.get(CANARY_PID) or {}).get("records", []))
    save_json(can, f"{base}/canary_score_{arm}.json")

    n = max(1, stats.get("total", 0))
    ovf = stats.get("overflow", 0)
    rates = {
        "first_pass": round(stats.get("first_pass", 0) / n, 3),
        "repair": round(stats.get("repaired", 0) / n, 3),
        "drop": round(stats.get("dropped", 0) / n, 3),
        "residual_overflow": round(ovf / max(1, n + ovf), 3),
    }
    kinds = Counter(r["kind"] for p in checked.values() for r in p["records"])
    gates = {
        "first_pass>=0.60": rates["first_pass"] >= 0.60,
        "repair<=0.25": rates["repair"] <= 0.25,
        "drop<=0.15": rates["drop"] <= 0.15,
        "residual<=0.10": rates["residual_overflow"] <= 0.10,
        "canary_recall>=4": sum(can["facts"].values()) >= 4,
        "canary_traps==0": can["trap_fired"] == 0,
    }
    report = {"arm": arm, "model": model, "rates": rates, "stats": stats,
              "canary": can, "kind_distribution": dict(kinds),
              "timing_s": timing, "arm_wall_s": round(time.time() - t_arm, 1),
              "single_arm_gates": gates,
              "single_arm_pass": all(gates.values())}
    save_json(report, f"{base}/smoke_report_{arm}.json")
    print(f"\n=== ARM {arm} ({model}) ===\n{json.dumps(report, ensure_ascii=False, indent=1)}",
          flush=True)
    return report


def judge_arm(arm: str, base: str, texts_dir: str):
    texts = dict(load_corpus(texts_dir))
    texts[CANARY_PID] = CANARY_TEXT
    checked = load_json(f"{base}/records_checked_{arm}.json", {})
    pool = [(pid, r) for pid, p in checked.items() for r in p["records"]]
    rng = random.Random(42)  # deterministic sample across arms
    by_kind = {}
    for pid, r in pool:
        by_kind.setdefault(r["kind"], []).append((pid, r))
    # proportional stratified sample of exactly 10 (v1 bug: 1-per-kind gave 6)
    kinds = sorted(by_kind)
    total = max(1, len(pool))
    alloc = {k: max(1, round(10 * len(by_kind[k]) / total)) for k in kinds}
    while sum(alloc.values()) > 10:
        k = max(alloc, key=lambda x: alloc[x])
        alloc[k] -= 1
    sample = []
    for k in kinds:
        sample += rng.sample(by_kind[k], min(alloc[k], len(by_kind[k])))
    sample = sample[:10]
    from kb_infra.llm import call_paratera, parse_json_response
    judged = []
    for pid, r in sample:
        text = texts.get(pid, "")
        loc = r.get("loc") or {}
        s = max(0, (loc.get("char_start") or 0) - 600)
        window = text[s:s + 2200] or text[:2200]
        prompt = (JUDGE_PROMPT.replace("{record}", json.dumps(r, ensure_ascii=False)[:2500])
                  .replace("{window}", window))
        # Kimi via bare call_paratera (proven judging path; NO enable_thinking
        # param — unverified on Kimi, old stack never sent it)
        obj = None
        for _ in range(3):
            obj = parse_json_response(
                call_paratera(prompt, model="Kimi-K2.6", max_tokens=400))
            if obj is not None:
                break
        if isinstance(obj, list):  # Kimi occasionally returns arrays — normalize
            obj = next((x for x in obj if isinstance(x, dict)), None)
        if not isinstance(obj, dict):
            judged.append({"paper_id": pid, "record_id": r.get("id"), "judge": None})
            continue
        vals = [obj.get("quote"), obj.get("binding"), obj.get("condition")]
        mean = sum(v for v in vals if isinstance(v, (int, float))) / max(1, len(vals))
        judged.append({"paper_id": pid, "record_id": r.get("id"), "kind": r["kind"],
                       "scores": obj, "mean": round(mean, 2)})
        print(f"  [{arm}|{pid}|{r['kind']}] {obj} ", flush=True)
    ok = [j["mean"] for j in judged if j.get("mean")]
    out = {"arm": arm, "judged": judged,
           "mean": round(sum(ok) / len(ok), 2) if ok else None, "n": len(ok)}
    save_json(out, f"{base}/smoke_judge_{arm}.json")
    return out


def final_gate(base: str):
    rd = load_json(f"{base}/smoke_report_dsf.json", {})
    rq = load_json(f"{base}/smoke_report_qwen.json", {})
    jd = load_json(f"{base}/smoke_judge_dsf.json", {})
    jq = load_json(f"{base}/smoke_judge_qwen.json", {})
    if not (rd and rq):
        # single-arm run (v3+): model selection already resolved by v2 A/B
        # verdict (DSF retired per pre-registered fallback gate)
        if rq:
            verdict = {
                "qwen_single_arm": rq["single_arm_pass"],
                "ab_gates": None,
                "model_selection": "Qwen3.8-Max (resolved at smoke v2; DSF retired)",
                "decision": ("Qwen 单臂全闸通过" if rq["single_arm_pass"]
                             else "Qwen 单臂仍有红闸（管线继续迭代，门柱不软）"),
            }
            save_json(verdict, f"{base}/smoke_verdict.json")
            print("\n=== FINAL GATE (single-arm) ===\n"
                  + json.dumps(verdict, ensure_ascii=False, indent=1), flush=True)
        else:
            print("final gate: missing arm reports", flush=True)
        return
    ab = {
        "first_pass_gap_ok": rd["rates"]["first_pass"] >= rq["rates"]["first_pass"] - 0.10,
        "traps_ok": rd["canary"]["trap_fired"] <= rq["canary"]["trap_fired"],
        "recall_ok": sum(rd["canary"]["facts"].values()) >=
                     sum(rq["canary"]["facts"].values()) - 1,
        "drop_gap_ok": rd["rates"]["drop"] <= rq["rates"]["drop"] + 0.10,
    }
    if jd.get("mean") is not None and jq.get("mean") is not None:
        ab["judge_mean_ok"] = jd["mean"] >= 3.5 and jd["mean"] >= jq["mean"] - 0.5
    verdict = {
        "dsf_single_arm": rd["single_arm_pass"], "qwen_single_arm": rq["single_arm_pass"],
        "ab_gates": ab,
        "decision": ("DSF合格(低成本档保留)" if (rd["single_arm_pass"] and all(ab.values()))
                     else "回退Qwen3.8-Max(按预注册)"),
    }
    save_json(verdict, f"{base}/smoke_verdict.json")
    print("\n=== FINAL GATE ===\n" + json.dumps(verdict, ensure_ascii=False, indent=1),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["extract", "judge", "all"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--arms", default="dsf,qwen")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",")]
    if args.stage in ("extract", "all"):
        for arm in arms:
            extract_arm(arm, ARMS[arm], args.base, args.texts)
    if args.stage in ("judge", "all"):
        for arm in arms:
            judge_arm(arm, args.base, args.texts)
        final_gate(args.base)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
