# -*- coding: utf-8 -*-
"""生长马拉松 v2（10-01：更长时程+可量化生长对比，论文素材级）。

v1（growth_demo.py，12轮）遗留三病与 v2 对策：
  1. 选缺口撞偏词（MNIST/mixup 单token subject→检索太泛，3/12 轮
     no_relevant_paper）→ v2: 缺口打分（subject+missing 词法丰度+
     类型优先级），偏词缺口自然沉底
  2. 检索词只 2 条（synthesize_queries 直出）→ v2: 并 3 条
     （词法变体扩展复用 findings 的形态学规则思想）
  3. 矩阵/桥实体增长要终态手工重编译才可见 → v2: 跑完自动
     pre/post 重编译对比（矩阵表数、跨论文可比表、桥实体数、
     resolved 缺口数——生长账本一次出齐）

循环逻辑与 v1 相同（缺口→检索词→检索→挑论文→admit+deep_read→
backflow→resolved 复检→溯源账本），24 轮。

用法：PYTHONUTF8=1 python growth_marathon.py [n_rounds=24]
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
DEMO = os.path.join(CS2, "demo_growth_kb2")
sys.path.insert(0, SRC)
sys.path.insert(0, SHARED)

# 挂死蔓延修复(2026-10-01 马拉松 r6-r24 实锤):默认 LLM_WALL_TIMEOUT=240s
# ×4 次 retry=僵尸线程最长 16 分钟持锁;看门狗 detach 的僵尸继续 retry
# 循环占 LOCAL 信号量车道,后续 deep_read 连环饿死(24 轮 22 轮空转)。
# 马拉松场景收紧:墙钟 90s(实测正常调用 1-141s,慢调用多为排队异常)
# +retry=1(僵尸最多 90s 释放车道)。独立进程复现 deep_read 全通证明
# 代码本体无 bug,此为进程内状态蔓延的对症处置。
os.environ.setdefault("LLM_WALL_TIMEOUT", "90")
os.environ.setdefault("LOCAL_MAX_ATTEMPTS", "1")
os.environ.setdefault("LOCAL_SOCK_TIMEOUT", "60")

from kb_infra.llm import call_paratera  # noqa: E402
from retrieval.gap_search import synthesize_queries  # noqa: E402

MODEL = "DeepSeek-V4.1-Flash"
READER = "local:Qwen3.8-27B"

_STOP = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "to",
         "not", "was", "is", "are", "with", "by", "et", "al"}


def _norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def setup_demo_kb():
    os.makedirs(DEMO, exist_ok=True)
    import shutil
    # 全新快照（v2 从干净起点开始——v1 的 demo_growth_kb 是旧 KB 副本）
    for name in ("views_cs2.json", "registry_v2.json", "manifest_all.json",
                 "backflow_edges.jsonl", "deep_read_records.json"):
        src = os.path.join(BASE_KB, name)
        dst = os.path.join(DEMO, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print(f"[marathon] KB snapshot at {DEMO}")


def load_gap_pool():
    views = json.load(open(os.path.join(DEMO, "views_cs2.json"),
                           encoding="utf-8"))
    ab = (views.get("coverage") or {}).get("absences_extracted") or []
    prio = {"explicitly_stated": 0, "not_reported": 1,
            "survey_claimed": 2, "cannot_tell": 3}
    pool = [a for a in ab if not a.get("resolved_by")]
    # v2 病1治疗：词法丰度打分——subject/missing 的有效 token 数
    # （单 token subject 的缺口检索面太泛自然沉底；类型优先级仍主导）
    def _richness(a):
        subj_toks = [t for t in _norm(a.get("subject")).split()
                     if t not in _STOP]
        miss_toks = [t for t in _norm(a.get("missing")).split()
                     if t not in _STOP]
        return len(subj_toks) * 2 + min(len(miss_toks), 8)
    pool.sort(key=lambda a: (prio.get(a.get("absence_type"), 9),
                             -_richness(a)))
    return views, pool


def expand_queries(queries):
    """v2 病2治疗：检索词并集扩展（缺口的确定性变体：去限定词、
    主动语态改名词形态的粗规则 + synthesize_queries 原生 2-3 条）。"""
    out = list(queries)
    for q in queries[:2]:
        toks = [t for t in q.split() if t.lower() not in _STOP]
        if len(toks) >= 2:
            v = " ".join(toks)
            if v not in out:
                out.append(v)
    return out[:4]


def pick_paper(cands, gap):
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
    """P0-2 同款核验（v1 修正版：全字段拼接+缺口相关排序窗口）。"""
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


def compile_metrics(records_path, registry_path, label):
    """生长度量：矩阵表数/跨论文可比表/桥实体数。"""
    import subprocess
    # 重编译到临时 views
    tmp_views = os.path.join(DEMO, f"views_{label}.json")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = SRC
    r = subprocess.run(
        [sys.executable, "-m", "kb_compiler.views.compiler",
         "--records", records_path, "--registry", registry_path,
         "--vocab", os.path.join(BASE_KB, "dim_vocab_cs2.json"),
         "--manifest", os.path.join(DEMO, "manifest_all.json"),
         "--out", tmp_views],
        capture_output=True, text=True, env=env, encoding="utf-8",
        errors="replace")
    if r.returncode != 0:
        print(f"[metrics] compile FAILED: {r.stderr[-300:]}")
        return None
    v = json.load(open(tmp_views, encoding="utf-8"))
    tables = v.get("matrix", {}).get("tables", {})
    n_tables = len(tables)
    n_cross = sum(
        1 for ents in tables.values()
        if any(len({c.get("paper_id") for c in cells}) >= 2
               for cells in ents.values()))
    # 桥实体：records 里实体载体跨≥2论文
    reg = json.load(open(registry_path, encoding="utf-8"))
    byid = {e["entity_id"]: e for e in reg.get("entities", [])}
    REFF = ("method_ref", "from_method_ref", "to_method_ref",
            "scope_ref_ref", "target_ref_ref")
    ent2pids = {}
    d = json.load(open(records_path, encoding="utf-8"))
    for pid, payload in d.items():
        if not isinstance(payload, dict):
            continue
        for rec in payload.get("records") or []:
            eids = set()
            for f in REFF:
                ref = rec.get(f)
                if isinstance(ref, dict) and ref.get("entity_id"):
                    eids.add(ref["entity_id"])
            for a in rec.get("entity_refs") or []:
                if isinstance(a, dict) and a.get("entity_id"):
                    eids.add(a["entity_id"])
            for e in eids:
                ent2pids.setdefault(e, set()).add(pid)
    n_bridge = sum(1 for p in ent2pids.values() if len(p) >= 2)
    return {"matrix_tables": n_tables, "cross_paper_tables": n_cross,
            "bridge_entities": n_bridge}


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    setup_demo_kb()
    views, pool = load_gap_pool()
    tried = set()
    # 旧 demo 试过的缺口不再重复
    old_ledger = os.path.join(CS2, "growth_demo_ledger.json")
    if os.path.exists(old_ledger):
        tried |= {e.get("gap_record_id")
                  for e in json.load(open(old_ledger, encoding="utf-8"))}
    ledger = []
    print(f"[marathon] gap pool: {len(pool)} unresolved "
          f"({len(tried)} already tried in v1); running {n_rounds} rounds")

    # 生长前基线度量
    pre_records = os.path.join(DEMO, "records_pre.json")
    merged = json.load(open(os.path.join(BASE_KB, "records_merged.json"),
                            encoding="utf-8"))
    deep0 = json.load(open(os.path.join(DEMO, "deep_read_records.json"),
                           encoding="utf-8"))
    # 合并基线（demo 起点=base+已有深读——与 attach 时的 records 视图一致）
    base_records = {}
    for pid, p in merged.items():
        if isinstance(p, dict) and p.get("records"):
            base_records[pid] = {"records": list(p["records"])}
    for pid, p in deep0.items():
        if isinstance(p, dict) and p.get("records"):
            b = (base_records.get(pid) or {}).get("records") or []
            seen = {r.get("id") for r in b}
            base_records[pid] = {"records": b + [
                r for r in p["records"] if r.get("id") not in seen]}
    json.dump(base_records, open(pre_records, "w", encoding="utf-8"),
              ensure_ascii=False)
    m_pre = compile_metrics(pre_records, os.path.join(DEMO, "registry_v2.json"), "pre")
    print(f"[marathon] PRE metrics: {m_pre}")

    os.environ["KB_OPEN_SET"] = "1"
    sys.path.insert(0, CS2)
    from external_tools import attach_external_tools
    from kb_compiler.views.tools import KBTools
    manifest = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest.json"), encoding="utf-8"))}
    registry = json.load(open(os.path.join(DEMO, "registry_v2.json"),
                              encoding="utf-8"))
    vocab = {"subject": [], "setup": [], "variant": [], "hyperparam_items": []}
    records = {pid: dict(p) for pid, p in base_records.items()}
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
    # 断点续跑:跑过的缺口(v2 ledger 已有)不重复
    led_path = os.path.join(CS2, "growth_marathon_ledger.json")
    if os.path.exists(led_path):
        old = json.load(open(led_path, encoding="utf-8"))
        tried |= {e.get("gap_record_id") for e in old.get("ledger", [])}
        print(f"[marathon] resume: {len(old.get('ledger', []))} rounds "
              f"already done in v2 ledger")

    def _search_round(queries):
        """检索子步单独跑(看门狗内)。"""
        cands, seen_t = [], set()
        for q in queries[:3]:
            res = kb._ext_tools.search_papers(q, k=8)
            for p in (res.get("papers") or []):
                t = p.get("title")
                if t and t not in seen_t:
                    seen_t.add(t)
                    cands.append(p)
        return cands

    import concurrent.futures as _cf
    WATCHDOG_S = 420   # 单轮子步看门狗:7min(正常检索<1min,LLM 挑选<30s)

    def _wd(fn, desc):
        """子步看门狗:挂死子步按 None 处理,轮记 watchdog_timeout。
        py-spy 实锤教训(2026-10-01):with-block 的 __exit__→shutdown 会
        join 卡死的工作线程(27B getresponse 挂起+llm._timed_open 内部
        join 传染)——看门狗自己被拖死。修:不 shutdown,executor 泄漏
        一个线程换取主循环存活(daemon 线程进程退出时自动收)。"""
        ex = _cf.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=WATCHDOG_S)
        except _cf.TimeoutError:
            print(f"[marathon] WATCHDOG: {desc} hung >{WATCHDOG_S}s, "
                  f"skipping (thread left detached)", flush=True)
            return None
        except Exception as e:
            print(f"[marathon] {desc} error: {str(e)[:80]}", flush=True)
            return None

    for rnd in range(1, n_rounds + 1):
        gap = None
        for a in pool:
            if a.get("record_id") not in tried:
                tried.add(a.get("record_id"))
                if a.get("subject") and a.get("missing"):
                    gap = a
                    break
        if gap is None:
            print("[marathon] gap pool exhausted")
            break
        entry = {"round": rnd, "gap_record_id": gap.get("record_id"),
                 "gap_subject": gap.get("subject"),
                 "gap_missing": str(gap.get("missing"))[:200],
                 "gap_type": gap.get("absence_type")}
        queries = expand_queries(synthesize_queries(gap))
        entry["queries"] = queries
        cands = _wd(lambda: _search_round(queries), f"r{rnd} search")
        if cands is None:
            entry["outcome"] = "watchdog_timeout_search"
            ledger.append(entry)
            continue
        entry["n_candidates"] = len(cands)
        title = _wd(lambda: pick_paper(cands, gap), f"r{rnd} pick")
        if not title:
            entry["outcome"] = "no_relevant_paper"
            ledger.append(entry)
            print(f"[r{rnd}] {str(gap.get('subject'))[:40]}: "
                  f"no relevant paper among {len(cands)}", flush=True)
            continue
        entry["paper"] = title
        adm = kb._ext_tools.admit_paper(title=title)
        pid = adm.get("paper_id")
        if not pid:
            entry["outcome"] = f"admit_failed: {str(adm)[:80]}"
            ledger.append(entry)
            print(f"[r{rnd}] {str(gap.get('subject'))[:40]}: admit failed",
                  flush=True)
            continue
        dr = _wd(lambda: kb._ext_tools.deep_read(pid), f"r{rnd} deep_read")
        deep = json.load(open(os.path.join(DEMO, "deep_read_records.json"),
                              encoding="utf-8"))
        new_recs = (deep.get(pid) or {}).get("records") or []
        entry["paper_id"] = pid
        entry["n_new_records"] = len(new_recs)
        resolved = _wd(lambda: check_resolution(gap, new_recs), f"r{rnd} verify") \
            if new_recs else False
        if resolved is None:
            resolved = False
        entry["resolved"] = resolved
        entry["outcome"] = ("resolved" if resolved else "ingested_not_resolving")
        ledger.append(entry)
        print(f"[r{rnd}] {str(gap.get('subject'))[:38]} <- {title[:46]} "
              f"({len(new_recs)} recs, resolved={resolved})", flush=True)

    # 终态：合并新深读 → 生长度量
    dt = time.time() - t0
    deep = json.load(open(os.path.join(DEMO, "deep_read_records.json"),
                          encoding="utf-8"))
    final_records = {pid: dict(p) for pid, p in base_records.items()}
    for pid, p in deep.items():
        if isinstance(p, dict) and p.get("records"):
            b = (final_records.get(pid) or {}).get("records") or []
            seen = {r.get("id") for r in b}
            final_records[pid] = {"records": b + [
                r for r in p["records"] if r.get("id") not in seen]}
    post_records = os.path.join(DEMO, "records_final.json")
    json.dump(final_records, open(post_records, "w", encoding="utf-8"),
              ensure_ascii=False)
    m_post = compile_metrics(post_records, os.path.join(DEMO, "registry_v2.json"), "post")

    n_resolved = sum(1 for e in ledger if e.get("resolved"))
    n_ingested = sum(1 for e in ledger if e.get("paper_id"))
    n_new_papers = len(set(k for k in deep) - set(deep0))
    n_new_records = sum(len((deep.get(k) or {}).get("records") or [])
                        - len((deep0.get(k) or {}).get("records") or [])
                        for k in deep if k in deep0) + \
        sum(len((deep.get(k) or {}).get("records") or []) for k in deep
            if k not in deep0)
    bf_lines = sum(1 for l in open(os.path.join(DEMO, "backflow_edges.jsonl"),
                                   encoding="utf-8") if l.strip()) \
        if os.path.exists(os.path.join(DEMO, "backflow_edges.jsonl")) else 0

    print(f"\n[marathon] DONE in {dt/60:.0f} min")
    print(f"  rounds={len(ledger)} ingested={n_ingested} "
          f"no_relevant={sum(1 for e in ledger if e.get('outcome')=='no_relevant_paper')} "
          f"resolved={n_resolved}")
    print(f"  new papers={n_new_papers} new records≈{n_new_records} "
          f"backflow_lines={bf_lines}")
    print(f"  PRE : {m_pre}")
    print(f"  POST: {m_post}")
    if m_pre and m_post:
        delta = {k: m_post[k] - m_pre[k] for k in m_pre}
        print(f"  Δ   : {delta}")
    out = {"rounds": len(ledger), "ingested": n_ingested,
           "resolved": n_resolved, "new_papers": n_new_papers,
           "new_records": n_new_records, "backflow_lines": bf_lines,
           "minutes": round(dt / 60, 1), "pre": m_pre, "post": m_post,
           "ledger": ledger}
    json.dump(out, open(os.path.join(CS2, "growth_marathon_ledger.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[marathon] ledger -> growth_marathon_ledger.json")


if __name__ == "__main__":
    main()
