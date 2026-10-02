# -*- coding: utf-8 -*-
"""组件4验收:also_discussed_in + papers_discussing_entity。"""
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
deep = json.load(open(base + "/deep_read_records.json", encoding="utf-8"))
for pid, payload in deep.items():
    base_recs = (records.get(pid) or {}).get("records") or []
    seen = {r.get("id") for r in base_recs if r.get("id")}
    records[pid] = {"records": base_recs + [
        r for r in payload.get("records", []) if r.get("id") not in seen]}
kb = KBTools(views, registry, vocab, manifest, records)

# 模拟 31a 病例:agent 拿 novel-ideas 论文查 findings(实际调用形态:
# paper_id + contains,无 entity)
ff = [p for p in records if "unlock_novel" in p][0]
print(f"查询论文: {ff[:60]}")
res = kb.findings(paper_id=ff, contains="diversity|novelty|homogenization", k=10)
print(f"findings: n={res['n']}")
print("related_papers_via_entities:")
for a in res.get("related_papers_via_entities") or []:
    print(f"   {a['paper_id'][:55]} records_on_entity={a['records_on_entity']}")

print()
res2 = kb.find_gap(entity="bert")
print(f"find_gap(bert): absences={len(res2['absences_extracted'])}")
print("papers_discussing_entity:")
for a in res2.get("papers_discussing_entity") or []:
    print(f"   {a['paper_id'][:55]} records_on_entity={a['records_on_entity']}")
