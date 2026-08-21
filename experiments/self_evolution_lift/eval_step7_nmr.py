# -*- coding: utf-8 -*-
"""step7 模块1: NMR (Node Match Ratio) evaluator — Intern-Atlas style.

Match induced method nodes (LLM-named, e.g. "μ(I) rheology") to gold method
ids (M1..M26) via GLM-Embedding-2 semantic similarity + Hungarian assignment.
NMR = fraction of gold method nodes matched by an induced node (cosine >= thresh).

This is the technical core of module-1 survey-gold evaluation: induced names
are free-form, gold is M-id + method-name desc, so we need semantic matching
(Intern-Atlas uses the same idea) instead of string equality.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from granular_agent.llm_client import embed_batch

# gold method id -> canonical method-name description (mined from GOLD_RELS desc)
GOLD_METHODS = {
    "M1":  "μ(I) rheology / local viscoplastic constitutive law (inertial number)",
    "M17": "I-gradient model / nonlocal gradient expansion of μ(I) rheology",
    "M10": "kinetic theory of rapid granular flow",
    "M11": "extended kinetic theory / dense-flow extension of kinetic theory",
    "M21": "Yoon-Jenkins dense inclined flow kinetic extension",
    "M23": "Gray particle-size segregation shallow-layer theory",
    "M26": "Tripathi density-difference-driven segregation model",
    "M6":  "kinematic model / void-diffusion segregation",
    "M7":  "drainage / void model",
    "M8":  "spot model for random-packing dynamics",
}
# gold relations (from eval_lift_14papers)
GOLD_RELS = [
    {"src": "M17", "type": "extends",  "tgt": "M1"},
    {"src": "M11", "type": "extends",  "tgt": "M10"},
    {"src": "M26", "type": "improves", "tgt": "M23"},
    {"src": "M21", "type": "extends",  "tgt": "M10"},
    {"src": "M8",  "type": "compares",  "tgt": "M6"},
    {"src": "M7",  "type": "compares",  "tgt": "M6"},
]

THRESH = 0.55  # cosine threshold for a match (tunable; GLM-Embedding-2)


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b)) / (
        (sum(x*x for x in a)**0.5) * (sum(y*y for y in b)**0.5) + 1e-9)


def hungarian(cost):
    """Max-weight bipartite assignment (greedy; fine for ~10x10). cost[i][j]=score."""
    n, m = len(cost), len(cost[0]) if cost else 0
    pairs = []
    used_j = set()
    for i in range(n):
        best_j, best_v = -1, -1
        for j in range(m):
            if j not in used_j and cost[i][j] > best_v:
                best_j, best_v = j, cost[i][j]
        if best_j >= 0 and best_v >= 0:
            pairs.append((i, best_j, best_v))
            used_j.add(best_j)
    return pairs


def main():
    # induced method names (from a real lift run — step7_ablation arm T cluster output)
    # these come from the lift; we hardcode the names seen in prior runs for this test.
    induced = [
        "μ(I) rheology",
        "nonlocal granular fluidity model",
        "动理学理论 (Kinetic theory of rapid granular flow)",
        "动理学理论的密堆扩展 (Dense extension of kinetic theory)",
        "颗粒尺寸分离的浅层理论 (Gray segregation shallow-layer)",
        "Tripathi density-difference-driven segregation",
        "运动学模型 (Kinematic / void diffusion)",
        "Spot Model (random-packing dynamics)",
    ]
    print(f"induced methods: {len(induced)}", flush=True)
    print(f"gold methods: {len(GOLD_METHODS)}", flush=True)

    # embed both
    gold_ids = list(GOLD_METHODS.keys())
    gold_names = [GOLD_METHODS[m] for m in gold_ids]
    embs = embed_batch(induced + gold_names)
    if any(e == [0.0]*1024 or max(e)==0 for e in embs):
        print("[WARN] some embeddings zeroed (API issue)")
    ind_emb = embs[:len(induced)]
    gold_emb = embs[len(induced):]

    # cost[i][j] = cosine(induced_i, gold_j)
    cost = [[cosine(ind_emb[i], gold_emb[j]) for j in range(len(gold_ids))]
            for i in range(len(induced))]
    pairs = hungarian(cost)

    matched_gold = set()
    print("\n=== matches (induced -> gold, cosine) ===")
    for i, j, v in pairs:
        if v >= THRESH:
            matched_gold.add(gold_ids[j])
            print(f"  {induced[i][:45]:<45} -> {gold_ids[j]}  cos={v:.3f}")
        else:
            print(f"  {induced[i][:45]:<45} -> {gold_ids[j]}  cos={v:.3f} (below thresh {THRESH})")

    nmr = len(matched_gold) / len(gold_ids)
    print(f"\nNMR (node match ratio) = {nmr:.3f}  ({len(matched_gold)}/{len(gold_ids)} gold matched)")
    print(f"matched gold: {sorted(matched_gold)}")
    print(f"unmatched gold: {sorted(set(gold_ids)-matched_gold)}")

    # ERR: how many gold relations have both endpoints matched + a relation lifted between them
    # (coarse — full ERR needs the lifted relation graph, which varies per arm)
    print("\n=== gold relation endpoint coverage (for ERR precheck) ===")
    for g in GOLD_RELS:
        s_ok = g['src'] in matched_gold
        t_ok = g['tgt'] in matched_gold
        print(f"  {g['src']}({s_ok}) --{g['type']}--> {g['tgt']}({t_ok}): {'both matched' if s_ok and t_ok else 'partial/unmatched'}")


if __name__ == "__main__":
    main()
