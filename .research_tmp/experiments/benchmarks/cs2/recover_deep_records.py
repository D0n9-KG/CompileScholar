# -*- coding: utf-8 -*-
"""从 deep_read_cache.jsonl 离线恢复终化记录（deterministic finalize）。

批 10 的 deep_read 记录只活在答题进程内存——进程退出后适配器的
EvidenceStore 解析不了深记录回指（[0b3fa784...] 不在 records_merged）。
finalize 是 (text, card, chunk objs) 的确定性函数：缓存里三样俱全，
重跑 finalize 得到与进程内完全一致的记录 id。产物落
base_kb/deep_read_records.json（适配器叠加层）。

用法：PYTHONUTF8=1 python recover_deep_records.py
"""
import json
import os
import sys

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
sys.path.insert(0, SRC)

from kb_compiler.records.deep_extract import (  # noqa: E402
    build_paper_tasks, finalize_paper)

import hashlib


def main():
    cache_lines = [json.loads(l) for l in open(
        os.path.join(BASE_KB, "deep_read_cache.jsonl"), encoding="utf-8")
        if l.strip()]
    by_paper = {}
    for e in cache_lines:
        ent = by_paper.setdefault(e["paper_id"], {"text_hash": None,
                                                  "card": None, "chunks": {}})
        if e.get("text_hash"):
            ent["text_hash"] = e["text_hash"]
        if e.get("type") == "card":
            ent["card"] = e.get("card")
        elif e.get("chunk_id"):
            ent["chunks"][e["chunk_id"]] = e.get("obj")

    out = {}
    for pid, ent in by_paper.items():
        if not ent["card"]:
            continue
        text_path = os.path.join(
            BASE_KB, "deep_read_texts",
            hashlib.md5(pid.encode()).hexdigest() + ".txt")
        if not os.path.exists(text_path):
            continue
        text = open(text_path, encoding="utf-8").read()
        th = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        if ent["text_hash"] and th != ent["text_hash"]:
            print(f"[skip] {pid[:45]}: text hash mismatch")
            continue
        title = pid
        # manifest title if available
        try:
            manifest = {r["paper_id"]: r for r in json.load(
                open(os.path.join(BASE_KB, "manifest_all.json"),
                     encoding="utf-8"))}
            title = (manifest.get(pid) or {}).get("title") or pid
        except Exception:
            pass
        vocab = {"subject": [], "setup": [], "variant": [],
                 "hyperparam_items": []}
        seed_reg = {"entities": [], "surface_index": {}}
        ctx = build_paper_tasks(pid, text, ent["card"], seed_reg, vocab,
                                title)
        chunk_objs = {ch["chunk_id"]: ent["chunks"].get(ch["chunk_id"])
                      for ch, _ in ctx["chunk_prompts"]}
        absence_obj = ent["chunks"].get(f"{pid}#absence")
        # 全 chunk 都已抽（L2 完成过的论文）才终化全量；否则只终化已抽
        # 子集（与进程内 L1 视图一致——L2 未完的记录本来就没入过账）
        have = {cid for cid, o in chunk_objs.items() if o is not None}
        if f"{pid}#absence" in have and len(have) >= len(chunk_objs):
            ctx_full = ctx
        else:
            sel = [(ch, p) for ch, p in ctx["chunk_prompts"]
                   if ch["chunk_id"] in have]
            ctx_full = {"inj": ctx["inj"], "chunks": [c for c, _ in sel],
                        "chunk_prompts": sel,
                        "absence_prompt": ctx["absence_prompt"]}
            chunk_objs = {ch["chunk_id"]: ent["chunks"].get(ch["chunk_id"])
                          for ch, _ in sel}
            absence_obj = None
        _, out_payload = finalize_paper(pid, ctx_full, chunk_objs,
                                        absence_obj, seed_reg, ent["card"],
                                        title)
        out[pid] = out_payload
        n = len(out_payload.get("records", []))
        print(f"[ok] {pid[:50]}: {n} records")

    outp = os.path.join(BASE_KB, "deep_read_records.json")
    json.dump(out, open(outp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"saved {len(out)} papers -> {outp}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
