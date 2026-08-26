# -*- coding: utf-8 -*-
"""A2 core metrics: cross-paper pattern adoption + evolution activity.

Compares an evolution-arm bundle set vs a frozen-arm set (same papers).
Metrics (pre-registered in A2_granular_arm_design.md):
  1. cross-paper pattern reuse: how many edges in paper N use patterns that
     were ADDED by papers 1..N-1 (not seed patterns) — THE in-loop metric
  2. evolution triggers: patterns added per paper, cumulative curve
  3. pattern survival: added patterns still active at the end
Usage: python .research_tmp/A2_analysis.py EVO|FROZEN
Reads bundles .research_tmp/runs/kernel_v2/{A2G_EVO|A2G_FRZ}_*/  (the run_kernel
--tag layout) — the LAST bundle of each arm run holds the final shared KB, but
per-paper adoption needs the per-paper ledger progression; we reconstruct from
each paper's new_patterns.json + concept_graph.json.
"""
import json, os, sys, re

BASE = '.research_tmp/runs/kernel_v2'

SEED_PATTERNS = None  # filled from first bundle's meta (12-13 seed pats)


def load_bundle(tag):
    dirs = sorted(d for d in os.listdir(BASE) if d.startswith(f'{tag}_PPR'))
    return dirs


def main(tag):
    global SEED_PATTERNS
    bundles = load_bundle(tag)
    # paper order = run order; run_kernel names dirs {tag}_{ppr}; we need time order
    # — recover from each bundle's result.json timestamp
    entries = []
    for d in bundles:
        r = json.load(open(os.path.join(BASE, d, 'result.json'), encoding='utf-8'))
        entries.append((r.get('timestamp', ''), d, r))
    entries.sort()
    print(f"{tag}: {len(entries)} papers in run order:")
    seed_pats = set(json.load(open(os.path.join(BASE, entries[0][1], 'meta_snapshot.json'), encoding='utf-8'))['patterns'].keys()) if entries else set()

    added_so_far = set()
    total_reused_edges = 0
    total_edges = 0
    for ts, d, r in entries:
        np = json.load(open(os.path.join(BASE, d, 'new_patterns.json'), encoding='utf-8'))
        new_this = [p for p in np if p not in seed_pats]
        cg = json.load(open(os.path.join(BASE, d, 'concept_graph.json'), encoding='utf-8'))
        # edges from THIS paper: provenance paper_id == this paper
        ppr = re.search(r'(PPR_[A-Z0-9]+)', d).group(1)
        mine = [he for he in cg['hyperedges']
                if any(p.get('paper_id') == ppr for p in he.get('provenance', []))]
        reused = [he for he in mine if he.get('pattern_type') in added_so_far]
        total_edges += len(mine)
        total_reused_edges += len(reused)
        print(f"  {d[:26]:28s} ts={ts[:19]} edges={len(mine):3d} "
              f"new_pats={len(new_this):2d} reused_prior={len(reused):2d} "
              f"({', '.join(sorted(set(he['pattern_type'] for he in reused))[:4]) if reused else '-'})")
        added_so_far |= set(new_this)
    print(f"\nCUMULATIVE: patterns added across run: {len(added_so_far)}")
    print(f"CROSS-PAPER REUSE: {total_reused_edges}/{total_edges} edges "
          f"({total_reused_edges/total_edges*100:.1f}%)" if total_edges else "no edges")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'A2G_EVO')
