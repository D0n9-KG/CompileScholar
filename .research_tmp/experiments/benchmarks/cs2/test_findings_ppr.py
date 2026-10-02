# -*- coding: utf-8 -*-
"""组件1验收:findings() PPR 重排对粗抽记录生效(CS2_PPR=1)。

修复前:粗抽记录无 scope/target → _rec_ppr 恒 0 → 永远落 lo 桶;
修复后:entity_refs 绑定实体参与 PPR 计分 → 图邻域粗抽记录进 hi 桶。
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.views.tools import KBTools  # noqa: E402

os.environ["CS2_PPR"] = "1"

base = "base_kb"
views = json.load(open(base + "/views_cs2.json", encoding="utf-8"))
registry = json.load(open(base + "/registry_v2.json", encoding="utf-8"))
vocab = json.load(open(base + "/dim_vocab_cs2.json", encoding="utf-8"))
manifest = {r["paper_id"]: r for r in json.load(
    open(base + "/manifest.json", encoding="utf-8"))}
records = json.load(open(base + "/records_merged.json", encoding="utf-8"))
deep = json.load(open(base + "/deep_read_records.json", encoding="utf-8"))
for pid, payload in deep.items():  # runner 同款叠加
    base_recs = (records.get(pid) or {}).get("records") or []
    seen = {r.get("id") for r in base_recs if r.get("id")}
    records[pid] = {"records": base_recs + [
        r for r in payload.get("records", []) if r.get("id") not in seen]}
kb = KBTools(views, registry, vocab, manifest, records)

res = kb.findings(entity="bert", contains="performance", k=40)
print(f"findings(bert, performance): n={res['n']} returned={res['returned']} "
      f"ppr_reranked={res.get('ppr_reranked')}")
hi_boundary = None
for i, e in enumerate(res["entries"]):
    tag = ""
    if e.get("paper_id", "").startswith(("sciverse_", "arxiv_")):
        tag = "[非hub=粗抽/survey]"
    print(f"  {i:2d}. {e['paper_id'][:45]:47s} "
          f"{str(e.get('claim'))[:60]!r} {tag}")
    if i > 17:
        break
