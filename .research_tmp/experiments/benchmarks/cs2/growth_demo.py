# -*- coding: utf-8 -*-
"""生长演示跑（FIX-PLAN v2 §3.3 载定：缺口驱动自主生长循环，无题参与）。

叙事四件套的完整闭环演示：知缺（缺口池）→定向探索（缺口→检索词）→
会生长（admit+deep_read+backflow 入库）→可追溯（resolved_by 生灭+全程
溯源日志）。形态=离线演示跑：初始 KB 快照 → N 轮循环 → 终态+溯源账本。

循环（每轮）：
  1. 从缺口池选一个未尝试缺口（优先 explicitly_stated/not_reported——
     survey_claimed 是综述转述次之；排序=类型优先级+subject 实体可解析）
  2. synthesize_queries(缺口) → 定向检索词（缺口文本贡献题面没有的词汇）
  3. search_papers(检索词) → 候选；LLM 挑 1 篇最对症的（缺口匹配）
  4. admit_paper + deep_read（真入库）
  5. backflow 自动挂谱系（P0-1 修复的机制）
  6. resolved_by 复检：新记录是否实质解决该缺口（P0-2 同款 LLM 核验）
     → 解决=缺口标记 resolved（生灭）；未解决=记录尝试，缺口保留
  7. 全程溯源日志：每篇入库论文 ← 哪条缺口 ← 哪个检索词 ← 哪次检索

度量（跑完输出）：
  a. 生长行为：入库论文数/新增记录数/新增谱系边/backflow 命中实体数
  b. 缺口解决率：尝试的缺口中 resolved 比例
  c. 矩阵覆盖对比：前后 matrix tables 数（生长效果的自然量化指标）

反作弊形态：无题参与、题序无关；每步可溯源。KB 写入走独立演示目录
（demo_growth_kb/ 快照副本）——不污染 base_kb（答题模式用）。

用法：PYTHONUTF8=1 python growth_demo.py [n_rounds=12]
"""
import json
import os
import re
import sys
import time

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
SHARED = os.path.normpath(os.path.join(CS2, "..", "_shared", "tools"))
DEMO = os.path.join(CS2, "demo_growth_kb")
sys.path.insert(0, SRC)
sys.path.insert(0, SHARED)

from kb_infra.llm import call_paratera  # noqa: E402
from retrieval.gap_search import synthesize_queries  # noqa: E402

MODEL = "DeepSeek-V4.1-Flash"      # 编排/核验（便宜）
READER = "local:Qwen3.8-27B"       # 深读抽取（与答题臂同源）


def _norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def setup_demo_kb():
    """演示 KB=base_kb 快照副本（写入走这里，正本不动）。"""
    os.makedirs(DEMO, exist_ok=True)
    for name in ("views_cs2.json", "registry_v2.json", "manifest_all.json",
                 "backflow_edges.jsonl", "deep_read_records.json"):
        src = os.path.join(BASE_KB, name)
        dst = os.path.join(DEMO, name)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
    print(f"[demo] KB snapshot at {DEMO}")


def load_gap_pool():
    """缺口池（带类型优先级）+ 已 resolved 排除。"""
    views = json.load(open(os.path.join(DEMO, "views_cs2.json"),
                           encoding="utf-8"))
    ab = (views.get("coverage") or {}).get("absences_extracted") or []
    prio = {"explicitly_stated": 0, "not_reported": 1,
            "survey_claimed": 2, "cannot_tell": 3}
    pool = [a for a in ab if not a.get("resolved_by")]
    pool.sort(key=lambda a: prio.get(a.get("absence_type"), 9))
    return views, pool


def pick_gap(pool, tried):
    """选缺口：类型优先级序里第一个未尝试且 subject 可读的。"""
    for a in pool:
        key = a.get("record_id")
        if key in tried:
            continue
        if a.get("subject") and a.get("missing"):
            return a
    return None


def pick_paper(cands, gap):
    """LLM 挑最对症的一篇（缺口→候选匹配）。返回 title 或 None。"""
    if not cands:
        return None
    listing = "\n".join(
        f"{i+1}. {c.get('title')} ({c.get('year')}) — "
        f"{str(c.get('abstract') or '')[:200]}"
        for i, c in enumerate(cands[:8]))
    prompt = f"""A knowledge base records this research gap:
  Subject: {gap.get('subject')}
  Missing: {str(gap.get('missing'))[:400]}

Here are candidate papers from a search:
{listing}

Which ONE paper most plausibly CONTRIBUTES TO FILLING this gap (proposes, evaluates, or surveys exactly the missing capability)? Answer with the number alone (or 0 if none is relevant)."""
    try:
        out = call_paratera(prompt, model=MODEL, max_tokens=10,
                            temperature=0.0, enable_thinking=False)
        m = re.search(r"\d+", out or "")
        i = int(m.group()) if m else 0
        return cands[i - 1].get("title") if 1 <= i <= len(cands[:8]) else None
    except Exception:
        return None


def check_resolution(gap, new_records):
    """P0-2 同款核验：新记录是否实质解决缺口。
    冒烟实测修正×2（r1 误判两次）：① config/result 记录 claim 空——
    全字段拼接；② 截断窗口 [0:14] 对"解决证据在哪"盲（VGG 的 jitter
    记录在 14/15 位被切掉一格）——先按缺口 token 重叠排序再取窗口。"""
    def _rec_text(r):
        parts = [str(r.get(k) or "") for k in
                 ("claim", "subject", "item", "value", "measure",
                  "condition", "missing")]
        return " ".join(p for p in parts if p and p != "None")[:160]

    gap_toks = set(_norm(str(gap.get("subject")) + " " +
                         str(gap.get("missing"))).split())

    def _rel(r):
        return len(gap_toks & set(_norm(_rec_text(r)).split()))

    ranked = sorted(new_records, key=_rel, reverse=True)[:14]
    rec_s = "\n".join(
        f"- [{r.get('kind')}] {_rec_text(r)}"
        for r in ranked if _rec_text(r).strip())
    prompt = f"""A knowledge base recorded this gap:
  Subject: {gap.get('subject')}
  Missing: {str(gap.get('missing'))[:400]}

A newly ingested paper produced these records:
{rec_s}

Does the new paper plausibly RESOLVE the specific gap (provide what was missing)? First line exactly one word: YES or NO."""
    try:
        out = call_paratera(prompt, model=MODEL, max_tokens=120,
                            temperature=0.0, enable_thinking=False)
        return (out or "").strip().splitlines()[0].strip().upper().startswith("YES")
    except Exception:
        return False


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    setup_demo_kb()
    views, pool = load_gap_pool()
    tried = set()
    ledger = []          # 溯源账本：round→gap→queries→paper→records→resolved
    print(f"[demo] gap pool: {len(pool)} unresolved gaps; "
          f"running {n_rounds} rounds")

    # 挂外部工具（对演示 KB 副本操作——写入全落 demo 目录）
    os.environ["KB_OPEN_SET"] = "1"
    sys.path.insert(0, CS2)
    from cs2_runner import build_tools_cs2 as _orig
    # 直接构造（绕开 runner 的 ARM 路径），路径指 demo 目录：
    from external_tools import attach_external_tools
    from kb_compiler.views.tools import KBTools
    records = {}
    merged = json.load(open(os.path.join(DEMO, "records_merged.json"),
                            encoding="utf-8")) if os.path.exists(
        os.path.join(DEMO, "records_merged.json")) else json.load(
        open(os.path.join(BASE_KB, "records_merged.json"), encoding="utf-8"))
    for pid_key, payload in merged.items():
        if isinstance(payload, dict) and payload.get("records"):
            records[pid_key] = payload
    manifest = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest.json"), encoding="utf-8"))}
    registry = json.load(open(os.path.join(DEMO, "registry_v2.json"),
                              encoding="utf-8"))
    vocab = {"subject": [], "setup": [], "variant": [], "hyperparam_items": []}
    kb = KBTools(views, registry, vocab, manifest, records)
    attach_external_tools(
        kb, views, manifest, model=READER, registry=registry,
        blocklist=None,
        backflow_path=os.path.join(DEMO, "backflow_edges.jsonl"),
        replay_from=os.path.join(DEMO, "backflow_edges.jsonl"),
        tier_db=os.path.join(DEMO, "growth_library.db"),
        manifest_path=os.path.join(DEMO, "manifest_all.json"),
        deep_cache_path=os.path.join(DEMO, "deep_read_cache.jsonl"),
        deep_text_dir=os.path.join(DEMO, "deep_read_texts"),
        deep_records_path=os.path.join(DEMO, "deep_read_records.json"))
    from kb_compiler.records.backflow import load_backflow
    load_backflow(views, os.path.join(DEMO, "backflow_edges.jsonl"))

    t0 = time.time()
    for rnd in range(1, n_rounds + 1):
        gap = pick_gap(pool, tried)
        if gap is None:
            print("[demo] gap pool exhausted")
            break
        tried.add(gap.get("record_id"))
        entry = {"round": rnd, "gap_record_id": gap.get("record_id"),
                 "gap_subject": gap.get("subject"),
                 "gap_missing": str(gap.get("missing"))[:200],
                 "gap_type": gap.get("absence_type")}
        # ① 缺口→检索词（定向探索）
        queries = synthesize_queries(gap)
        entry["queries"] = queries
        # ② 检索
        cands = []
        for q in queries[:2]:
            res = kb._ext_tools.search_papers(q, k=8)
            cands.extend(res.get("papers") or [])
        entry["n_candidates"] = len(cands)
        # ③ 挑最对症的一篇
        title = pick_paper(cands, gap)
        if not title:
            entry["outcome"] = "no_relevant_paper"
            ledger.append(entry)
            print(f"[r{rnd}] {str(gap.get('subject'))[:40]}: "
                  f"no relevant paper among {len(cands)}")
            continue
        entry["paper"] = title
        # ④ 入库（admit+deep_read——会生长）
        adm = kb._ext_tools.admit_paper(title=title)
        pid = adm.get("paper_id")
        if not pid:
            entry["outcome"] = f"admit_failed: {str(adm)[:80]}"
            ledger.append(entry)
            print(f"[r{rnd}] {str(gap.get('subject'))[:40]}: admit failed")
            continue
        dr = kb._ext_tools.deep_read(pid)
        n_recs = len(json.loads(
            open(os.path.join(DEMO, "deep_read_records.json"),
                 encoding="utf-8").read()).get(pid, {}).get("records") or []) \
            if os.path.exists(os.path.join(DEMO, "deep_read_records.json")) else 0
        entry["paper_id"] = pid
        entry["n_new_records"] = n_recs
        # ⑤ resolved_by 复检（可追溯）
        deep = json.load(open(os.path.join(DEMO, "deep_read_records.json"),
                              encoding="utf-8"))
        new_recs = (deep.get(pid) or {}).get("records") or []
        resolved = check_resolution(gap, new_recs) if new_recs else False
        entry["resolved"] = resolved
        entry["outcome"] = ("resolved" if resolved else "ingested_not_resolving")
        ledger.append(entry)
        print(f"[r{rnd}] {str(gap.get('subject'))[:40]} <- {title[:50]} "
              f"({n_recs} recs, resolved={resolved})", flush=True)
        # 视图重载（backflow 挂边累积进 demo views）
        views = json.load(open(os.path.join(DEMO, "views_cs2.json"),
                               encoding="utf-8")) if hasattr(kb, '_ext_tools') else views

    # 终态度量
    dt = time.time() - t0
    deep = json.load(open(os.path.join(DEMO, "deep_read_records.json"),
                          encoding="utf-8")) if os.path.exists(
        os.path.join(DEMO, "deep_read_records.json")) else {}
    n_new_papers = len(deep)
    n_new_records = sum(len((p or {}).get("records") or [])
                        for p in deep.values())
    bf_lines = sum(1 for l in open(os.path.join(DEMO, "backflow_edges.jsonl"),
                                   encoding="utf-8") if l.strip()) \
        if os.path.exists(os.path.join(DEMO, "backflow_edges.jsonl")) else 0
    n_resolved = sum(1 for e in ledger if e.get("resolved"))
    n_ingested = sum(1 for e in ledger if e.get("paper_id"))
    print(f"\n[demo] DONE in {dt/60:.0f} min: rounds={len(ledger)} "
          f"ingested_papers={n_ingested} new_records={n_new_records} "
          f"backflow_lines={bf_lines} gaps_resolved={n_resolved}")
    json.dump(ledger, open(os.path.join(CS2, "growth_demo_ledger.json"),
                           "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[demo] ledger -> growth_demo_ledger.json")


if __name__ == "__main__":
    main()
