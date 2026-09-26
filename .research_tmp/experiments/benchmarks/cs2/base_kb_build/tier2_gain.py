# -*- coding: utf-8 -*-
"""0.6 step 5: Tier-2 gain table + pool freeze (prereg PREREG-0.6 §二).

gain(P) = demand(P) × (1 + α·sel_neighbors(P)) × (1 + β·gap_signal(P))
  demand(P)        dev100 rehearsal hit score: rank r -> 1/r, summed
  sel_neighbors(P) greedy-updated count of already-selected papers sharing
                   >=2 normalized subjects/mentions with P
  gap_signal(P)    normalized Tier-1 limitation-record count
  α=β=0.5 (frozen)

Selection: greedy argmax until pool target; floor = top-12 demand families
each >=15 papers.

Subject/mention normalization (user-flagged fix): alias folding —
lowercase, strip parentheticals/abbrev expansions, merge known surface
variants (llms/large language models/...), singular/plural fold.

Usage: python tier2_gain.py [--target 300] [--floor-families 12]
Inputs: base_kb/{manifest.json, demand_rehearsal.json, coarse_records.json}
Output: base_kb/{tier2_gain_table.json, tier2_pool.json}
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_KB = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
               r"\experiments\benchmarks\cs2\base_kb")
ALPHA = 0.5
BETA = 0.5

# ---------------------------------------------------------- normalization

_ABBR_STRIP = re.compile(r"\s*\([^a-z0-9]{0,4}\)\s*")     # (LLM) short
_PAREN_STRIP = re.compile(r"\s*\([^)]*\)")                # any parenthetical
_PLURAL = re.compile(r"([a-z]+)s$")

ALIAS_GROUPS = [
    {"llm", "llms", "large language model", "large language models",
     "large language models llm", "language model", "language models"},
    {"transformer", "transformers", "transformer model",
     "transformer models", "transformer architecture"},
    {"neural network", "neural networks", "neural net", "neural nets",
     "deep neural network", "deep neural networks", "dnn", "dnns"},
    {"convolutional neural network", "convolutional neural networks",
     "cnn", "cnns", "convolutional network"},
    {"reinforcement learning", "rl"},
    {"generative adversarial network", "generative adversarial networks",
     "gan", "gans"},
    {"graph neural network", "graph neural networks", "gnn", "gnns"},
    {"fine-tuning", "finetuning", "fine tuning"},
    {"in-context learning", "icl", "in context learning"},
    {"machine learning", "ml"},
    {"deep learning", "dl"},
]


def _fold(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PAREN_STRIP.sub(" ", s)      # strip parentheticals (abbr/expansions)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = _PLURAL.match(s)
    if m and len(m.group(1)) > 3:
        s = m.group(1)
    return s


_ALIAS_MAP = {}
for _g in ALIAS_GROUPS:
    _canon = sorted(_g, key=len)[0]
    for _a in _g:
        _ALIAS_MAP[_a] = _canon


def norm_surface(s: str) -> str:
    f = _fold(s)
    return _ALIAS_MAP.get(f, f)


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--floor-families", type=int, default=12)
    ap.add_argument("--floor-per-family", type=int, default=15)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    manifest = json.load(open(BASE_KB / "manifest.json", encoding="utf-8"))
    rehearsal = json.load(open(BASE_KB / "demand_rehearsal.json",
                               encoding="utf-8"))
    coarse = json.load(open(BASE_KB / "coarse_records.json",
                            encoding="utf-8"))
    by_paper = {r["paper_id"]: r for r in manifest}

    # ---- title -> paper_id index first, then demand scores
    pid_by_title = {}
    for pid, r in by_paper.items():
        pid_by_title[_nt(r.get("title"))] = pid
    demand = Counter()
    for q in rehearsal:
        for rank, hit in enumerate(q.get("hits", []), 1):
            pid = pid_by_title.get(_nt(hit.get("title")))
            if pid:
                demand[pid] += 1.0 / rank
    print(f"[gain] demand-scored papers: {len(demand)}", flush=True)

    # ---- paper entity sets (normalized subjects+mentions) + gap signal
    ents: dict[str, set] = defaultdict(set)
    gaps: Counter = Counter()
    for pid_key, payload in coarse.items():
        # coarse_records keyed by manifest row key (paper_id or title key)
        pid = pid_by_title.get(_nt(pid_key), pid_key)
        for r in payload.get("records", []):
            if r.get("subject"):
                ents[pid].add(norm_surface(r["subject"]))
            for m in r.get("mentions", []):
                if m:
                    ents[pid].add(norm_surface(m))
            if r.get("kind") == "limitation":
                gaps[pid] += 1
    max_gap = max(gaps.values()) if gaps else 1
    print(f"[gain] entity sets: {len(ents)} papers | "
          f"gap carriers: {len(gaps)}", flush=True)

    # ---- greedy selection
    def sel_neighbors(pid, selected):
        return sum(1 for q in selected
                   if len(ents.get(pid, set()) & ents.get(q, set())) >= 2)

    def gap_norm(pid):
        return gaps.get(pid, 0) / max_gap

    scored = {}
    selected: list = []
    # candidates: papers with entity sets (coarse-extracted) — the pool
    # Tier-2 draws from; prioritize demand>0
    cands = sorted(ents.keys(),
                   key=lambda p: -demand.get(p, 0.0))
    # demand families for the floor rule: top families by aggregate demand
    fam_demand = Counter()
    for pid, d in demand.items():
        for e in ents.get(pid, set()):
            fam_demand[e] += d
    floor_fams = [e for e, _ in fam_demand.most_common(args.floor_families)]

    while len(selected) < args.target and cands:
        best, best_g = None, -1.0
        for pid in cands:
            if pid in selected:
                continue
            g = (demand.get(pid, 0.0) or 0.01) \
                * (1 + ALPHA * sel_neighbors(pid, selected)) \
                * (1 + BETA * gap_norm(pid))
            if g > best_g:
                best, best_g = pid, g
        if best is None:
            break
        scored[best] = best_g
        selected.append(best)
        cands.remove(best)
        if len(selected) % 50 == 0:
            print(f"  selected {len(selected)}/{args.target} "
                  f"(last gain {best_g:.3f})", flush=True)

    # ---- floor rule: top families each >=N papers
    fam_counts = Counter()
    for pid in selected:
        for e in ents.get(pid, set()):
            if e in floor_fams:
                fam_counts[e] += 1
    deficit = []
    for fam in floor_fams:
        if fam_counts.get(fam, 0) < args.floor_per_family:
            deficit.append((fam, fam_counts.get(fam, 0)))
    if deficit:
        print(f"[gain] floor deficits: {deficit}", flush=True)
        for fam, have in deficit:
            need = args.floor_per_family - have
            fam_cands = sorted(
                (p for p in ents
                 if fam in ents[p] and p not in selected),
                key=lambda p: (-demand.get(p, 0.0), -gaps.get(p, 0)))
            for p in fam_cands[:need]:
                selected.append(p)

    # ---- write gain table + pool
    table = []
    for pid in set(selected):
        table.append({
            "paper_id": pid,
            "title": (by_paper.get(pid) or {}).get("title"),
            "demand": round(demand.get(pid, 0.0), 4),
            "n_entities": len(ents.get(pid, set())),
            "gap_records": gaps.get(pid, 0),
            "gain_at_selection": round(scored.get(pid, 0.0), 4),
            "selected": True,
            "arxiv_id": (by_paper.get(pid) or {}).get("arxiv_id"),
        })
    # full non-selected stats for the archive (compact: top 500 by demand)
    for pid, d in demand.most_common(500):
        if pid not in selected:
            table.append({
                "paper_id": pid, "title": (by_paper.get(pid) or
                                           {}).get("title"),
                "demand": round(d, 4),
                "n_entities": len(ents.get(pid, set())),
                "gap_records": gaps.get(pid, 0),
                "selected": False,
                "arxiv_id": (by_paper.get(pid) or {}).get("arxiv_id"),
            })
    json.dump(table, open(BASE_KB / "tier2_gain_table.json", "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    pool = {
        "target": args.target,
        "alpha": ALPHA, "beta": BETA,
        "n_selected": len(selected),
        "n_with_arxiv_id": sum(1 for p in selected
                               if (by_paper.get(p) or {}).get("arxiv_id")),
        "papers": [p for p in selected],
    }
    json.dump(pool, open(BASE_KB / "tier2_pool.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[gain] pool frozen: {len(selected)} papers "
          f"({pool['n_with_arxiv_id']} with arxiv id)", flush=True)


def _nt(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()[:120]


if __name__ == "__main__":
    main()
