# -*- coding: utf-8 -*-
"""step7 模块4 消融全表: 5 arms over 19 papers.

arms:
  F  frozen        — no lift (meta has 0 METHOD_/higher_order)
  T  text-only     — lift_corpus(use_quals=False, paper_citations=None)
  C  +citation     — lift_corpus(use_quals=False, paper_citations=on)
  Q  +quals        — lift_corpus(use_quals=True,  paper_citations=None)
  B  +full         — lift_corpus(use_quals=True,  paper_citations=on)

metrics: families, relations, type distribution, gold type coverage.
Reuses cluster per arm (cheaper: share cluster? no — use_quals affects induce
which affects judge, so full re-lift per arm). Parallel judge keeps it feasible.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from collections import Counter
from granular_agent.hypergraph_schema import (
    seed_meta_hypergraph_general, MetaHypergraph,
)
from granular_agent.hypergraph_lifter import lift_corpus, METHOD_NS, HIGHER_ORDER_FAMILY

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "runs"), "ARFM2024", "full")
CIT = os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "runs"), "ARFM2024", "citations.json")
PAPERS = ["Midi_2004","Jop_2006","Pouliquen_1999_Scaling","Bouzid_2013","Bouzid_2015",
          "Kamrin_2012","Kamrin_2015","Jenkins_1983","Jenkins_2010","Gray_2005_A_theory",
          "Gray_2006_Particle-size","Savage_1988_Particle_size","Bazant_2006_The_spot",
          "Tüzün_1979_Experimental","Tripathi_2013_Density","Yoon_2006_The_influence",
          "Sarkar_2008_Experimental","Savage_1998_Analyses","Hill_2014_Segregation"]
GOLD_TYPES = ["extends","improves","compares"]
OUT = os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "runs"), "ARFM2024", "step7_ablation.json")


def find(pre):
    for f in os.listdir(ROOT):
        if f.startswith(pre) and f.endswith(".json"):
            return os.path.join(ROOT, f)
    return None


def build_ebp():
    ebp = {}
    for p in PAPERS:
        fp = find(p)
        if not fp:
            continue
        full = os.path.basename(fp)[:-5]
        d = json.load(open(fp, encoding='utf-8'))
        if d.get('edges'):
            ebp[full] = [{"paper": full, **e} for e in d['edges']]
    return ebp


def load_citations():
    d = json.load(open(CIT, encoding='utf-8'))
    return {n: list(rec.get("cites_corpus", [])) for n, rec in d.items()}


def frozen_stats(meta):
    """Count METHOD_ nodes + higher_order patterns in a fresh (never-lifted) schema."""
    mnodes = [t for t in meta.meta_nodes if t.startswith(METHOD_NS)]
    hrels = [p for p in meta.patterns
             if meta.patterns[p].family == HIGHER_ORDER_FAMILY]
    return {"families": 0, "relations": 0, "types": {},
            "gold_hit": 0, "method_nodes": len(mnodes),
            "higher_order_patterns": len(hrels)}


def run_arm(tag, ebp, pc, use_quals):
    print(f"\n=== arm {tag} (quals={'on' if use_quals else 'off'}, cite={'on' if pc else 'off'}) ===", flush=True)
    meta = seed_meta_hypergraph_general()
    out = lift_corpus(meta, ebp, paper_citations=pc, use_quals=use_quals)
    wr = out['written']['relation_patterns']
    by_type = Counter(r['relation'] for r in wr)
    types_present = set(by_type.keys())
    gold_hit = sum(1 for g in GOLD_TYPES if g in types_present)
    res = {"families": len(out['clusters']), "relations": len(wr),
           "types": dict(by_type), "gold_hit": gold_hit}
    print(f"families={res['families']} relations={res['relations']} "
          f"types={res['types']} gold={gold_hit}/3", flush=True)
    return res


def main():
    ebp = build_ebp()
    pc = load_citations()
    print(f"corpus: {len(ebp)} papers, {sum(len(v) for v in pc.values() if v)} citation edges", flush=True)

    arms = {}
    # F frozen
    print("\n=== arm F (frozen, no lift) ===", flush=True)
    arms["F_frozen"] = frozen_stats(seed_meta_hypergraph_general())
    print(f"  {arms['F_frozen']}", flush=True)
    # T text-only
    arms["T_text"] = run_arm("T text-only", ebp, None, use_quals=False)
    # C +citation
    arms["C_citation"] = run_arm("C +citation", ebp, pc, use_quals=False)
    # Q +quals
    arms["Q_quals"] = run_arm("Q +quals", ebp, None, use_quals=True)
    # B +full
    arms["B_full"] = run_arm("B +full", ebp, pc, use_quals=True)

    print(f"\n{'='*70}\n=== STEP7 ABLATION TABLE ===", flush=True)
    print(f"{'arm':<12} {'fam':<5} {'rel':<5} {'extends':<8} {'improves':<9} {'compares':<9} {'bg':<4} {'gold':<5}", flush=True)
    for tag, r in arms.items():
        t = r.get("types", {})
        print(f"{tag:<12} {r['families']:<5} {r['relations']:<5} "
              f"{t.get('extends',0):<8} {t.get('improves',0):<9} "
              f"{t.get('compares',0):<9} {t.get('background',0):<4} {r['gold_hit']}/3", flush=True)

    json.dump(arms, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
