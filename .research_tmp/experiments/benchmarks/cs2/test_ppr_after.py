# -*- coding: utf-8 -*-
"""组件1验收:PPR 传播(修复前恒空 → 修复后多跳)。"""
import json
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.views.tools import KBTools  # noqa: E402

base = "base_kb"
views = json.load(open(base + "/views_cs2.json", encoding="utf-8"))
registry = json.load(open(base + "/registry_v2.json", encoding="utf-8"))
vocab = json.load(open(base + "/dim_vocab_cs2.json", encoding="utf-8"))
manifest = {r["paper_id"]: r for r in json.load(
    open(base + "/manifest.json", encoding="utf-8"))}
records = json.load(open(base + "/records_merged.json", encoding="utf-8"))
kb = KBTools(views, registry, vocab, manifest, records)
byid = {e["entity_id"]: e for e in registry["entities"]}

for seed in ["transformer", "bert", "reinforcement learning"]:
    scores = kb.ppr_entity_scores([seed])
    print(f"\n=== 种子 {seed!r}: PPR 命中 {len(scores)} 实体 ===")
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:12]
    for eid, s in top:
        cn = byid.get(eid, {}).get("canonical", eid)
        print(f"  {s:.4f}  {cn!r}")
