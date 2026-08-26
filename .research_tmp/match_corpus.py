# -*- coding: utf-8 -*-
"""A2 granular arm prep: match gold method refs to corpus papers properly.

Author-surname-only matching is too loose (spot check: 2/3 wrong). Proper
signal: (a) first-author surname AND year in an early citation-looking block,
or (b) surname + a topic keyword from the method family. We validate by
checking the YEAR appears near the surname in the first ~20 blocks (source
papers cite themselves rarely, but MinerU headers carry the year for many
papers; better: match the paper's own title against known titles).

Honest approach: use DOI/known titles from the ARFM survey citation list.
The citations.json (built earlier for the +citation arm) has per-paper
references with DOIs — that's the reliable join key.
"""
import json, os, re, sys

GOLD = 'experiments/self_evolution_lift/gold/ARFM2024_gold.json'
CIT = 'experiments/self_evolution_lift/runs/ARFM2024/citations.json'  # if exists
BASE = "C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers"

g = json.load(open(GOLD, encoding='utf-8'))
refs = set()
for m in g['methods']:
    for r in m.get('refs', []):
        refs.add(r)

# Author-year -> structured
def parse_ref(r):
    m = re.match(r"(.+?)\s+(\d{4})", r)
    if not m:
        return None, None
    authors, year = m.group(1), m.group(2)
    first = re.split(r"[,&]| et al", authors)[0].strip()
    return first, year

# scan full text of every corpus paper for "Firstauthor ... year" within a
# citation context — too loose. Instead: scan for self-identifying signature:
# the paper's OWN author + year. MinerU headers often carry "Author et al.
# Title... journal year". We check the first 6 blocks for surname+year.
hits = {}
for ppr in os.listdir(BASE):
    try:
        cl = json.load(open(os.path.join(BASE, ppr, "content_list.json"), encoding="utf-8"))
    except Exception:
        continue
    head = " ".join(str(it.get("text", "")) for it in cl[:6])
    for r in refs:
        first, year = parse_ref(r)
        if not first:
            continue
        if first in head and year in head:
            hits.setdefault(r, []).append(ppr)

print(f"refs matched (surname+year in first blocks): {len(hits)}/{len(refs)}")
for r in sorted(hits):
    print(f"  {r}: {hits[r][:3]}")
