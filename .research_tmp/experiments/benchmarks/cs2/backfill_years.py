# -*- coding: utf-8 -*-
"""P0-3（FIX-PLAN v2）：谱系年份回填（盘上 views_cs2.json 一次性修复）。

问题：2406 条边 year 全 None、15100 节点 year 全 None（A10 修了编译器
字段名，但盘上 views 是旧编译产物）——as_of 时间切片全部失明。

回填口径（写进字段 provenance 便于审计）：
- 边 year = 证据论文（edge.paper_id）发表年（arxiv pid 前缀 or manifest）
  ——与 compiler.build_genealogy 的既有语义一致
- 节点 year = 库内最早出现年：该实体 surface 命中 records_merged 记录
  的（subject/mentions）所在论文的最小年份。覆盖 41%（6201/15100）；
  其余保持 None（外部实体库内无证据，诚实缺失优于编造）。
  registry 的 in_corpus_paper_id/origin_year_cited 全空（v2 构建产物），
  此为唯一可溯数据源。

幂等：重跑覆盖同值。写回 views_cs2.json（git 在案可回滚）。

用法：PYTHONUTF8=1 python backfill_years.py [--dry]
"""
import json
import os
import re
import sys

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")


def _norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _paper_year(pid, man, man_all, hub_man=None):
    m = re.match(r"arxiv_(\d{2})\d{2}\.", pid or "")
    if m:
        return 2000 + int(m.group(1))
    for src in (man, man_all, hub_man or {}):
        if pid in src and src[pid].get("year"):
            try:
                return int(src[pid]["year"])
            except (TypeError, ValueError):
                return None
    return None


def main():
    dry = "--dry" in sys.argv
    views = json.load(open(os.path.join(BASE_KB, "views_cs2.json"),
                           encoding="utf-8"))
    reg = json.load(open(os.path.join(BASE_KB, "registry_v2.json"),
                         encoding="utf-8"))
    man = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest.json"), encoding="utf-8"))}
    man_all = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest_all.json"), encoding="utf-8"))}
    hub_man = {}
    _hp = os.path.join(BASE_KB, "hub_manifest.json")
    if os.path.exists(_hp):
        hub_man = {r["paper_id"]: r for r in json.load(
            open(_hp, encoding="utf-8"))}
    merged = json.load(open(os.path.join(BASE_KB, "records_merged.json"),
                            encoding="utf-8"))

    # ① 边回填
    g = views.setdefault("genealogy", {})
    edges = g.setdefault("edges", [])
    n_edge = 0
    for e in edges:
        y = _paper_year(e.get("paper_id"), man, man_all, hub_man)
        if y is not None and e.get("year") is None:
            e["year"] = y
            e["year_provenance"] = "evidence_paper"
            n_edge += 1
    filled = sum(1 for e in edges if e.get("year") is not None)
    print(f"[edges] backfilled {n_edge}, with year now {filled}/{len(edges)}")

    # ② 节点回填：surface 命中 → 最早库内出现年
    s2e = {}
    for surf, eid in (reg.get("surface_index") or {}).items():
        n = _norm(surf)
        if n:
            s2e.setdefault(n, set()).add(str(eid))
    ent_years = {}
    for pid, payload in merged.items():
        y = _paper_year(pid, man, man_all, hub_man)
        if y is None:
            continue
        for r in (payload.get("records") or []):
            for s in [r.get("subject")] + list(r.get("mentions") or [])[:4]:
                ids = s2e.get(_norm(s))
                if ids and len(ids) == 1:
                    eid = next(iter(ids))
                    if eid not in ent_years or y < ent_years[eid]:
                        ent_years[eid] = y
                    break
    nodes = g.setdefault("nodes", {})
    n_node = 0
    for eid, y in ent_years.items():
        nd = nodes.get(eid)
        if nd is not None and nd.get("year") is None:
            nd["year"] = y
            nd["year_provenance"] = "earliest_corpus_mention"
            n_node += 1
    n_with = sum(1 for nd in nodes.values() if nd.get("year") is not None)
    print(f"[nodes] backfilled {n_node}, with year now {n_with}/{len(nodes)}")

    if not dry:
        tmp = os.path.join(BASE_KB, "views_cs2.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(views, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(BASE_KB, "views_cs2.json"))
        print("[write] views_cs2.json updated")
    else:
        print("[dry] no write")


if __name__ == "__main__":
    main()
