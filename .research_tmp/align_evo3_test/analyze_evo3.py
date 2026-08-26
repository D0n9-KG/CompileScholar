# -*- coding: utf-8 -*-
"""A2 granular-flow cross-paper concept alignment first audit (pure analysis, no LLM).

Input : .research_tmp/runs/kernel_v2/A2G_EVO3_PPR_866D169A4642/concept_graph.json
        (last-in-time bundle == cumulative shared A-box over all 10 papers)
Output: console metrics + cross-paper variant lists + suspected duplicate pairs
        (written to variants_dump.txt / dup_pairs_dump.txt for report building)
"""
import json
import os
import sys
import io
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CG = os.path.abspath(os.path.join(BASE, "runs", "kernel_v2", "A2G_EVO3_PPR_866D169A4642", "concept_graph.json"))
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "align_evo3_test"))

with open(CG, encoding="utf-8") as f:
    g = json.load(f)
concepts = g["concepts"]

# ---------- a. totals & type distribution ----------
total = len(concepts)
type_dist = Counter(c.get("type", "?") for c in concepts.values())
print("=== a. totals ===")
print(f"total concepts: {total}")
for t, n in type_dist.most_common():
    print(f"  {t}: {n} ({n/total*100:.1f}%)")

# ---------- b. cross-paper merge rate ----------
def npapers(c):
    return len({v.get("paper_id") for v in c.get("surface_variants", [])})

cross = [c for c in concepts.values() if npapers(c) >= 2]
cross_method = [c for c in cross if c.get("type") == "METHOD"]
n_method = type_dist.get("METHOD", 0)
print("\n=== b. cross-paper merge rate ===")
print(f"concepts with >=2 distinct paper_ids: {len(cross)}/{total} ({len(cross)/total*100:.1f}%)")
print(f"METHOD cross-paper: {len(cross_method)}/{n_method} ({len(cross_method)/n_method*100:.1f}%) if n_method>0 else n/a")

# distribution of #papers per concept
pp_dist = Counter(npapers(c) for c in concepts.values())
print("distribution of distinct-paper-count per concept:", dict(sorted(pp_dist.items())))

# ---------- c. surface-string variability among cross-paper concepts ----------
print("\n=== c. surface variability among cross-paper concepts ===")
multi_surface = []
for c in cross:
    surfaces = {v.get("surface", "") for v in c.get("surface_variants", [])}
    if len(surfaces) >= 2:
        multi_surface.append((c, surfaces))
print(f"cross-paper concepts with >=2 distinct surface strings: {len(multi_surface)}/{len(cross)}")
mm_method = [(c, s) for (c, s) in multi_surface if c.get("type") == "METHOD"]
print(f"  of which METHOD: {len(mm_method)}")

# ---------- d/e. suspected duplicate pairs (token-set Jaccard >= 0.6) ----------
def toks(s):
    return frozenset("".join(ch.lower() if ch.isalnum() or ch == " " else " " for ch in s.lower()).split())

def first_surface(c):
    sv = c.get("surface_variants", [])
    return sv[0].get("surface", "") if sv else ""

def jaccard(a, b):
    A, B = toks(a), toks(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def dup_pairs(items):
    pairs = []
    items = list(items)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            c1, c2 = items[i], items[j]
            s1, s2 = first_surface(c1), first_surface(c2)
            if not s1 or not s2:
                continue
            J = jaccard(s1, s2)
            if J >= 0.6:
                pairs.append((c1["concept_id"], s1, c2["concept_id"], s2, round(J, 2)))
    return pairs

methods = [c for c in concepts.values() if c.get("type") == "METHOD"]
others = [c for c in concepts.values() if c.get("type") != "METHOD"]
dp_method = dup_pairs(methods)
dp_other = dup_pairs(others)
print("\n=== d. suspected duplicate METHOD pairs (Jaccard>=0.6, distinct cids) ===")
print(f"count: {len(dp_method)}")
for p in dp_method:
    print(f"  {p[0]} '{p[1]}'  <->  {p[2]} '{p[3]}'  J={p[4]}")
print(f"\n=== e. suspected duplicate pairs among NON-METHOD types ===")
print(f"count: {len(dp_other)}")
for p in dp_other:
    print(f"  {p[0]} '{p[1]}'  <->  {p[2]} '{p[3]}'  J={p[4]}")

# ---------- dumps for report ----------
with open(os.path.join(OUT, "variants_dump.txt"), "w", encoding="utf-8") as f:
    f.write(f"total={total} cross={len(cross)} cross_method={len(cross_method)} multi_surface={len(multi_surface)}\n")
    f.write("type_dist=" + json.dumps(dict(type_dist), ensure_ascii=False) + "\n\n")
    f.write("## ALL cross-paper concepts, full variant lists\n")
    order = {"METHOD": 0}
    for c in sorted(cross, key=lambda c: (order.get(c.get("type"), 1), -npapers(c))):
        surfaces = []
        for v in c.get("surface_variants", []):
            surfaces.append(f"[{v.get('paper_id')}|{v.get('section','')}] {v.get('surface')}")
        f.write(f"\n{c['concept_id']} type={c.get('type')} papers={npapers(c)}\n")
        for s in surfaces:
            f.write(f"   {s}\n")

with open(os.path.join(OUT, "dup_pairs_dump.txt"), "w", encoding="utf-8") as f:
    f.write("## METHOD dup pairs\n")
    for p in dp_method:
        f.write(f"{p[0]} | {p[1]} | {p[2]} | {p[3]} | J={p[4]}\n")
    f.write("\n## NON-METHOD dup pairs\n")
    for p in dp_other:
        f.write(f"{p[0]} | {p[1]} | {p[2]} | {p[3]} | J={p[4]}\n")

# also: per-paper concept counts sanity (which papers contributed)
paper_contrib = Counter()
for c in concepts.values():
    for pid in {v.get("paper_id") for v in c.get("surface_variants", [])}:
        paper_contrib[pid] += 1
print("\nconcepts touched per paper (cumulative graph):")
for pid, n in paper_contrib.most_common():
    print(f"  {pid}: {n}")
