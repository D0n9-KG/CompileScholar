# -*- coding: utf-8 -*-
"""Acceptance gate ①: calibrate blocking threshold tau on LEGACY ground truth.

Ground truth = PS-53-era registry_v32_pilot.json (668 entities / 870 surfaces,
human-audited through the PaperScope campaign). Embed all surfaces with the
local qwen3-embedding-8b, then for a tau sweep measure:
  - pair recall: fraction of same-entity alias pairs that land in ONE block
    (the anti-fragmentation guarantee; cross-block pass only recovers what
    tau2 catches, so recall at tau1 with tau2 below it is the real bound)
  - contamination: fraction of different-entity surface pairs sharing a block
    (tolerable — the LLM still arbitrates — but monitors block purity)

No LLM calls. Usage: python calibrate_tau.py
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
import numpy as np  # noqa: E402

from kb_compiler.records import embed_block  # noqa: E402

LEGACY = (r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/"
          r"archive/paperscope_2026-09/paperscope_r2/registry_v32_pilot.json")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "embed_cache", "embed_legacy_calibration.json")
TAUS = [0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    reg = json.load(open(LEGACY, encoding="utf-8"))
    surfaces, gid = [], []
    for gi, e in enumerate(reg["entities"]):
        for a in e["aliases"]:
            surfaces.append(a)
            gid.append(gi)
    print(f"legacy: {len(reg['entities'])} entities, {len(surfaces)} surfaces "
          f"({sum(1 for e in reg['entities'] if len(e['aliases']) > 1)} multi-alias)")

    embs = embed_block.embed_items(surfaces, CACHE)
    gid = np.asarray(gid)
    ac_edges = embed_block.acronym_edges(surfaces)
    inc_edges = embed_block.inclusion_edges(surfaces)
    # co-mention groups from legacy mention_papers
    surf_papers = [set() for _ in surfaces]
    gi = 0
    for e in reg["entities"]:
        for a in e["aliases"]:
            surf_papers[gi] = set(e.get("mention_papers") or [])
            gi += 1
    pgroups = {}
    for i, ps in enumerate(surf_papers):
        for pp in sorted(ps)[:3]:
            pgroups.setdefault(pp, []).append(i)
    com_edges = embed_block.paper_neighbor_edges(embs, list(pgroups.values()), top_k=2)
    for nm, ed in (("acronym", ac_edges), ("inclusion", inc_edges), ("comention(recovery-only)", com_edges)):
        sm = sum(1 for i, j in ed if gid[i] == gid[j])
        print(f"{nm} edges: {len(ed)} (same-entity {sm}, {sm/max(len(ed),1):.2f})")
    # blocking = embed + acronym + inclusion; comention is recovery-pass only
    all_edges = ac_edges | inc_edges

    # same-entity pairs / diff-entity pairs (index-based, no n^2 materialization
    # at this scale: 870^2 = 757k — fine to materialize)
    sim = embs @ embs.T
    iu = np.triu_indices(len(surfaces), k=1)
    sims, same = sim[iu], (gid[iu[0]] == gid[iu[1]])
    print(f"pairs: {len(sims)} total, {same.sum()} same-entity\n")

    print(f"{'tau':>5} | {'same-pair recall (+acronym)':>26} | {'cross-pair contam':>17} | blocks")
    for tau in TAUS:
        adj = embed_block._neighbors_above(embs, tau)
        for i, j in all_edges:
            adj[i].append(j)
            adj[j].append(i)
        comps = embed_block._components(adj, len(surfaces))
        comp_of = np.zeros(len(surfaces), dtype=int)
        for ci, c in enumerate(comps):
            for i in c:
                comp_of[i] = ci
        # pair metrics via cosine threshold directly (blocking recall ==
        # same-block iff connected; for pairs, direct sim>=tau is the
        # necessary condition and connected-component the sufficient report)
        rec_direct = float(((sims >= tau) & same).sum() / max(same.sum(), 1))
        rec_comp = float((same & (comp_of[iu[0]] == comp_of[iu[1]])).sum()
                         / max(same.sum(), 1))
        contam = float(((comp_of[iu[0]] == comp_of[iu[1]]) & ~same).sum()
                       / max((~same).sum(), 1))
        print(f"{tau:>5.2f} | {rec_direct:>7.3f} /{rec_comp:>6.3f} | {contam:>17.4f} | {len(comps)}")

    # missed alias pairs at the default tau1=0.86 (for eyeballing)
    tau = 0.86
    missed = [(surfaces[iu[0][k]], surfaces[iu[1][k]], float(sims[k]))
              for k in range(len(sims)) if same[k] and sims[k] < tau]
    missed.sort(key=lambda x: x[2])
    print(f"\nsame-entity pairs missed at tau={tau}: {len(missed)}")
    for a, b, s in missed[:25]:
        print(f"  {s:.3f}  {a[:45]!r} <-> {b[:45]!r}")


if __name__ == "__main__":
    main()
