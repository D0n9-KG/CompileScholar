# -*- coding: utf-8 -*-
"""组件1验收:共享实体桥密度(归一前 vs 归一后)。

修复前实体桥只在 views.genealogy 的名字键边上隐式存在(且与 entity_id
空间断裂);修复后 = 全部载体(bound refs + mentions + entity_refs)。
这是组件4(共享实体论文群)的直接地基。
"""
import json
import re
from collections import defaultdict

REFF = ("method_ref", "from_method_ref", "to_method_ref",
        "scope_ref_ref", "target_ref_ref")
_norm_re = re.compile(r"\s+")


def norm(s):
    return _norm_re.sub(" ", str(s or "").strip().lower())


def bridge_map(path_refs, path_deep):
    """entity_id -> set(paper_id),全载体。"""
    ent2pids = defaultdict(set)
    for path in (path_refs, path_deep):
        d = json.load(open(path, encoding="utf-8"))
        for pid, payload in d.items():
            if not isinstance(payload, dict):
                continue
            for r in payload.get("records") or []:
                for f in REFF:
                    ref = r.get(f)
                    if isinstance(ref, dict) and ref.get("entity_id"):
                        ent2pids[ref["entity_id"]].add(pid)
                for a in r.get("entity_refs") or []:
                    if a.get("entity_id"):
                        ent2pids[a["entity_id"]].add(pid)
    return ent2pids


def summarize(ent2pids, label, n_papers):
    bridges = {e: p for e, p in ent2pids.items() if len(p) >= 2}
    sizes = sorted((len(p) for p in bridges.values()), reverse=True)
    print(f"{label}: 桥实体(≥2论文)={len(bridges)}/{len(ent2pids)}")
    if sizes:
        print(f"  桥大小分布: max={sizes[0]} p50={sizes[len(sizes)//2]} "
              f"覆盖论文对≈{sum(s*(s-1)//2 for s in sizes)}")
    return bridges


base = "base_kb"
after = bridge_map(f"{base}/records_merged.json", f"{base}/deep_read_records.json")
before = bridge_map(f"{base}/records_merged.bak_pre_normalize.json",
                    f"{base}/deep_read_records.bak_pre_normalize.json")
n_papers = len({pid for pid in json.load(open(f"{base}/records_merged.json", encoding="utf-8"))})
b_before = summarize(before, "归一前", n_papers)
b_after = summarize(after, "归一后", n_papers)

# ab3651c4 病例论文:agent 磨过的 5 篇 + 它们的桥现在通向哪
probe_papers = [p for p in json.load(open(f"{base}/records_merged.json", encoding="utf-8"))
                if any(k in p for k in ("friend_or_foe", "can_large_language_models_unloc",
                                        "guidance_for_researchers", "advancing_the_scientific",
                                        "exploring_the_change_in_scienti"))]
print(f"\n病例论文 {len(probe_papers)} 篇的桥接度:")
for p in probe_papers:
    linked = set()
    for e, pids in b_after.items():
        if p in pids:
            linked |= {q for q in pids if q != p}
    linked_before = set()
    for e, pids in b_before.items():
        if p in pids:
            linked_before |= {q for q in pids if q != p}
    print(f"  {p[:60]}")
    print(f"    归一前可达论文 {len(linked_before)} → 归一后 {len(linked)}")
