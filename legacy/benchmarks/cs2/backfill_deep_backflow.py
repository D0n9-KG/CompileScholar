# -*- coding: utf-8 -*-
"""P0-1（FIX-PLAN v2）：存量深读记录批量补回流。

背景：_backflow_deep_records 的 mentions 提取只认粗抽字段
（method/subject/...），深读契约字段（scope_ref_ref/method_ref/
target_ref_ref 字典）全 miss——66 篇深读 7,395 条记录只有 12 篇
（粗抽路径）挂上回流边，"会生长"机制 09-26 起静默断裂。
external_tools._record_mentions 已修（双契约提取）；本脚本对存量
deep_read_records.json 重放同一逻辑，把新边追加进
backflow_edges.jsonl（运行时 load_backflow 重放挂载，不碰
views_cs2.json）。

幂等：先重放现有 backflow_edges.jsonl 进 scratch views（与运行时
同路径），apply_backflow 按 (from,to) 对去重——重复运行零新增。

用法：PYTHONUTF8=1 python backfill_deep_backflow.py [--dry]
"""
import json
import os
import sys

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
SHARED = os.path.normpath(os.path.join(CS2, "..", "_shared", "tools"))
sys.path.insert(0, SRC)
sys.path.insert(0, SHARED)

from kb_compiler.records.backflow import (  # noqa: E402
    build_backflow, apply_backflow)
from external_tools import _record_mentions  # noqa: E402


def load_meta():
    """paper_id → {title, year, doi}。三层：manifest.json →
    manifest_all.json → deep_read_cache 卡片 _title_used / paper_id 兜底
    （ext_ 论文的 manifest 条目在历史会话未落盘）。"""
    meta = {}
    for f in ("manifest.json", "manifest_all.json"):
        for r in json.load(open(os.path.join(BASE_KB, f), encoding="utf-8")):
            meta.setdefault(r["paper_id"], {
                "title": r.get("title"), "year": r.get("year"),
                "doi": r.get("doi")})
    for line in open(os.path.join(BASE_KB, "deep_read_cache.jsonl"),
                     encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "card" and e.get("paper_id") not in meta:
            t = (e.get("card") or {}).get("_title_used")
            if t:
                meta[e["paper_id"]] = {"title": t, "year": None, "doi": None}
    return meta


def main():
    dry = "--dry" in sys.argv
    deep = json.load(open(os.path.join(BASE_KB, "deep_read_records.json"),
                          encoding="utf-8"))
    registry = json.load(open(os.path.join(BASE_KB, "registry_v2.json"),
                              encoding="utf-8"))
    bl_path = os.path.join(BASE_KB, "blocklist_keep.json")
    blocklist = (json.load(open(bl_path, encoding="utf-8"))
                 if os.path.exists(bl_path) else None)
    meta = load_meta()
    bf_path = os.path.join(BASE_KB, "backflow_edges.jsonl")

    # scratch views：先重放现有持久化行（运行时同路径），保证幂等去重
    scratch = {"genealogy": {"nodes": {}, "edges": []}}
    n_exist = 0
    for line in open(bf_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("node"):
            apply_backflow(scratch, rec, persist_path=None)
            n_exist += 1

    n_papers = n_matched = n_new_edges = n_new_nodes = 0
    misses = []
    for pid, ent in deep.items():
        recs = (ent or {}).get("records") or []
        if not recs:
            continue
        n_papers += 1
        m = meta.get(pid) or {"title": pid, "year": None, "doi": None}
        payload = {"records": [
            {"id": r.get("id"), "kind": r.get("kind"),
             "subject": r.get("subject"), "claim": r.get("claim"),
             "quote": r.get("quote"),
             "mentions": _record_mentions(r)}
            for r in recs if isinstance(r, dict)]}
        bf = build_backflow(payload,
                            {"title": m.get("title") or pid,
                             "year": m.get("year"), "doi": m.get("doi")},
                            registry, blocklist)
        if not bf.get("matches"):
            # 无一 mention 命中 registry——记录下来（缺口池/registry
            # 覆盖面的诚实账目，不是错误）
            sample = [x["mentions"][:2] for x in payload["records"][:3]
                      if x["mentions"]]
            misses.append((pid, sample))
            continue
        n_matched += 1
        res = apply_backflow(
            scratch, bf, persist_path=None if dry else bf_path)
        n_new_edges += res.get("attached") or 0
        if not res.get("already_present"):
            n_new_nodes += 1
        print(f"[backfill] {pid[:55]} -> "
              f"{', '.join(res.get('matched_entities') or [])[:90]} "
              f"(+{res.get('attached') or 0} edges)", flush=True)

    print(f"\n[summary] papers={n_papers} matched={n_matched} "
          f"new_edges={n_new_edges} new_nodes={n_new_nodes} "
          f"replayed_existing={n_exist} dry={dry}")
    print(f"[misses] {len(misses)} papers with zero registry hits:")
    for pid, sample in misses[:10]:
        print(f"  {pid[:60]}  mentions e.g. {sample}")


if __name__ == "__main__":
    main()
