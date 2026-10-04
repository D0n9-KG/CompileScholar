# -*- coding: utf-8 -*-
"""夜间终局完整性检查。"""
import json
import os

base = "base_kb"
for f in ["records_merged.json", "deep_read_records.json",
          "views_cs2.json", "registry_v2.json"]:
    d = json.load(open(f"{base}/{f}", encoding="utf-8"))
    n = len(d) if isinstance(d, (dict, list)) else "?"
    print(f"{f}: OK ({n} entries)")
merged = json.load(open(f"{base}/records_merged.json", encoding="utf-8"))
n_er = sum(1 for p in merged.values() if isinstance(p, dict)
           for r in p.get("records", []) if r.get("entity_refs"))
n_ap = sum(1 for p in merged.values() if isinstance(p, dict)
           for r in p.get("records", []) if r.get("about_paper_id"))
n_pr = sum(1 for p in merged.values() if isinstance(p, dict)
           for r in p.get("records", []) if r.get("paraphrase_rel"))
print(f"entity_refs: {n_er} 记录 | about_paper: {n_ap} | paraphrase_rel: {n_pr}")
for bak in ["records_merged.bak_pre_normalize.json",
            "deep_read_records.bak_pre_normalize.json",
            "records_merged.bak_pre_citebridge.json",
            "views_cs2.bak_pre_normalize.json"]:
    ok = "OK" if os.path.exists(f"{base}/{bak}") else "MISSING"
    print(f"{bak}: {ok}")
