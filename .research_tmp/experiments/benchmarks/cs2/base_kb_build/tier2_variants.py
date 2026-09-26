# -*- coding: utf-8 -*-
"""Tier-2 pool variants comparison (user directive: 都试验一下).

V0 = original prereg formula (already run): demand × (1+.5n) × (1+.5g)
V1 = demand-squared:    demand^2 × (1+0.3n) × (1+0.5g)
V2 = two-stage:         top-150 by demand直接入池 -> 后 150 按原公式选
                        （族密度在第二阶段才生效）

Metrics (fixed BEFORE running, per user-approved criteria):
  m1 demand_top100_coverage   池内含 demand top-100 的篇数
  m2 demand_mass_coverage     池内 demand 总分 / 全库 demand>0 论文 demand 总分
  m3 family_density           池内平均同族邻居数（>=2 共享实体的池内邻居）
  m4 fulltext_availability    arxiv_id 或 doc_id 占比
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
            r"\experiments\benchmarks\cs2\base_kb")
sys.path.insert(0, str(Path(__file__).parent))
from tier2_gain import norm_surface  # noqa: E402  reuse alias folding

TARGET = 300


def _nt(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()[:120]


def load():
    manifest = json.load(open(BASE / "manifest.json", encoding="utf-8"))
    rehearsal = json.load(open(BASE / "demand_rehearsal.json",
                               encoding="utf-8"))
    coarse = json.load(open(BASE / "coarse_records.json", encoding="utf-8"))
    by_paper = {r["paper_id"]: r for r in manifest}
    pid_by_title = {_nt(r.get("title")): pid for pid, r in by_paper.items()}
    # coarse_extract CLI 的键 = title[:40]（截断）——补一个前缀索引桥接
    pid_by_title40 = {_nt((r.get("title") or "")[:40]): pid
                      for pid, r in by_paper.items()}
    demand = Counter()
    for q in rehearsal:
        for rank, hit in enumerate(q.get("hits", []), 1):
            pid = pid_by_title.get(_nt(hit.get("title")))
            if pid:
                demand[pid] += 1.0 / rank
    ents = defaultdict(set)
    gaps = Counter()
    unresolved = 0
    for pid_key, payload in coarse.items():
        pid = pid_by_title.get(_nt(pid_key)) \
            or pid_by_title40.get(_nt(pid_key)) or pid_key
        if pid not in by_paper:
            unresolved += 1
        for r in payload.get("records", []):
            if r.get("subject"):
                ents[pid].add(norm_surface(r["subject"]))
            for m in r.get("mentions", []):
                if m:
                    ents[pid].add(norm_surface(m))
            if r.get("kind") == "limitation":
                gaps[pid] += 1
    print(f"[load] coarse keys unresolved to manifest: {unresolved}",
          flush=True)
    return by_paper, demand, ents, gaps


def sel_neighbors(pid, selected, ents):
    return sum(1 for q in selected
               if len(ents.get(pid, set()) & ents.get(q, set())) >= 2)


def greedy(demand, ents, gaps, target, demand_pow=1.0,
           alpha=0.5, beta=0.5, first_stage_n=0):
    cands = sorted(ents.keys(), key=lambda p: -demand.get(p, 0.0))
    selected = []
    max_gap = max(gaps.values()) if gaps else 1
    # two-stage: first stage = pure demand order
    if first_stage_n:
        for pid in cands:
            if len(selected) >= first_stage_n:
                break
            if demand.get(pid, 0) > 0:
                selected.append(pid)
        cands = [p for p in cands if p not in selected]
    while len(selected) < target and cands:
        best, best_g = None, -1.0
        for pid in cands:
            d = demand.get(pid, 0.0) ** demand_pow
            if d <= 0:
                d = 0.001 ** demand_pow if demand_pow != 2 else 1e-6
            g = d * (1 + alpha * sel_neighbors(pid, selected, ents)) \
                * (1 + beta * gaps.get(pid, 0) / max_gap)
            if g > best_g:
                best, best_g = pid, g
        if best is None:
            break
        selected.append(best)
        cands.remove(best)
    return selected


def evaluate(selected, by_paper, demand, ents):
    top100 = [p for p, _ in demand.most_common(100)]
    m1 = sum(1 for p in top100 if p in selected)
    total_mass = sum(demand.values())
    pool_mass = sum(demand.get(p, 0) for p in selected)
    fam = [sel_neighbors(p, selected, ents) for p in selected]
    avail = sum(1 for p in selected
                if (by_paper.get(p) or {}).get("arxiv_id")
                or (by_paper.get(p) or {}).get("doc_id"))
    return {
        "m1_top100": m1,
        "m2_demand_mass": round(pool_mass / total_mass, 3) if total_mass else 0,
        "m3_family_density": round(sum(fam) / max(1, len(fam)), 1),
        "m4_fulltext": f"{avail}/{len(selected)}",
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    by_paper, demand, ents, gaps = load()
    variants = {
        "V0-original": dict(target=TARGET),
        "V1-demand-sq": dict(target=TARGET, demand_pow=2.0,
                             alpha=0.3, beta=0.5),
        "V2-two-stage": dict(target=TARGET, first_stage_n=150),
    }
    results = {}
    for name, kw in variants.items():
        print(f"[{name}] running greedy...", flush=True)
        sel = greedy(demand, ents, gaps, **kw)
        m = evaluate(sel, by_paper, demand, ents)
        results[name] = {"metrics": m, "pool": sel}
        print(f"  {m}", flush=True)
    json.dump({k: {"metrics": v["metrics"], "pool": v["pool"]}
               for k, v in results.items()},
              open(BASE / "tier2_variants_comparison.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\ncomparison saved -> tier2_variants_comparison.json")


if __name__ == "__main__":
    main()
