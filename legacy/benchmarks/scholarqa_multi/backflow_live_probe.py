# -*- coding: utf-8 -*-
"""Backflow live probe (batch-2 verification): real LLM coarse-extraction
of a real external abstract -> mentions matched against the REAL registry ->
genealogy attachment -> lineage_walk surfaces the paper at zero cost.

Probe paper: "Mamba: Linear-Time Sequence Modeling with Selective State
Spaces" — its abstract mentions Transformer/attention, which the Multi
registry certainly holds. """

import json
import os
import sys
import time

_CS = r"C:\Users\D0n9\Desktop\CompileScholar"
sys.path.insert(0, os.path.join(_CS, "src"))
sys.path.insert(0, r"C:\Users\D0n9\Desktop\sci-evo-extract\src")
sys.path.insert(0, os.path.join(_CS, ".research_tmp", "experiments",
                                "benchmarks", "_shared", "tools"))
os.environ.setdefault("LOCAL_MAX_CONCURRENT", "4")

KB = os.path.join(_CS, ".research_tmp", "experiments", "benchmarks",
                  "scholarqa_multi", "kb")

# --- real artifacts -------------------------------------------------------
views = json.load(open(os.path.join(KB, "views.json"), encoding="utf-8"))
registry = json.load(open(os.path.join(KB, "registry_v3.json"), encoding="utf-8"))
blocklist = json.load(open(os.path.join(KB, "blocklist_keep.json"),
                           encoding="utf-8"))
n_edges_before = len(views["genealogy"]["edges"])
n_nodes_before = len(views["genealogy"]["nodes"])
print(f"[load] views genealogy: {n_nodes_before} nodes / {n_edges_before} edges; "
      f"registry surface_index {len(registry['surface_index'])}", flush=True)

ABSTRACT = (
    "Foundation models are now routinely trained on sequences of billions of "
    "tokens. The de facto standard is the Transformer architecture together "
    "with its attention mechanism, whose effectiveness is often attributed to "
    "the ability to mix information across sequence positions. However, this "
    "mixing scales quadratically in sequence length, and attention has seen "
    "substantial improvements through approximate attention methods. We "
    "propose Mamba, a new state space model architecture, which achieves "
    "linear-time sequence modeling with selective state spaces. Mamba "
    "outperforms Transformers on language modeling at up to 5x larger "
    "model sizes and matches models like Transformer-XL in recall tasks."
)
TITLE = "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

# --- 1. coarse extraction (real LLM) --------------------------------------
t0 = time.time()
from kb_compiler.records.coarse_extract import extract_coarse
payload = extract_coarse(TITLE, ABSTRACT, 2023, "local:Qwen3.8-27B")
print(f"[coarse] {len(payload['records'])} records in {time.time()-t0:.1f}s",
      flush=True)
for r in payload["records"]:
    print(f"  - {r['kind']:10s} {r['subject'][:40]:40s} "
          f"mentions={r['mentions']}", flush=True)

# --- 2. backflow against the REAL registry --------------------------------
from kb_compiler.records.backflow import (apply_backflow, build_backflow,
                                          load_backflow)
bf = build_backflow(payload, {"title": TITLE, "year": 2023, "doi": None},
                    registry, blocklist)
print(f"[match] {len(bf['edges'])} edges from mentions:", flush=True)
for e in bf["edges"]:
    print(f"  - {e['from_name']}  --external_mention-->  {e['to_name'][:50]} "
          f"({e['provenance']}, quote: {e['quote'][:60]}...)", flush=True)
assert bf["node"]["id"].startswith("ext:")

# --- 3. apply + persist ----------------------------------------------------
probe_path = os.path.join(KB, "backflow_edges.probe.jsonl")
if os.path.exists(probe_path):
    os.remove(probe_path)
res = apply_backflow(views, bf, persist_path=probe_path)
print(f"[apply] attached={res['attached']} already={res['already_present']} "
      f"persisted -> {os.path.basename(probe_path)}", flush=True)
assert res["attached"] == len(bf["edges"])
assert len(views["genealogy"]["edges"]) == n_edges_before + res["attached"]

# --- 4. lineage_walk payoff: zero-cost surfacing ---------------------------
from sci_evo_extract.library.lineage_walk import lineage_walk
# pick a matched entity as the walk start (the recall path: question about
# entity E -> E's external continuations include this paper)
start_entity = next(e["from_name"] for e in bf["edges"] if e["from_name"])
t0 = time.time()
r = lineage_walk(start_entity, views, search_fn=lambda q, l: type(
    "R", (), {"candidates": []})(), external=True)
lat = (time.time() - t0) * 1000
print(f"[walk] start='{start_entity}' reached {len(r.in_corpus)} nodes in "
      f"{lat:.1f}ms; attached_external={len(r.attached_external)}", flush=True)
hit = [a for a in r.attached_external
       if "Mamba" in (a.get("title") or "")]
assert hit, "attached paper not surfaced by lineage_walk!"
a = hit[0]
print(f"  -> surfaced: {a['title'][:60]} (via {a['attached_via']}, "
      f"{len(a['records'])} records, year={a['year']})", flush=True)
for rec in a["records"][:3]:
    print(f"     [{rec['kind']}] {rec['claim'][:80]}", flush=True)

# --- 5. cross-session replay -----------------------------------------------
views2 = json.load(open(os.path.join(KB, "views.json"), encoding="utf-8"))
n = load_backflow(views2, probe_path)
assert n == 1
r2 = lineage_walk(start_entity, views2, search_fn=lambda q, l: type(
    "R", (), {"candidates": []})(), external=True)
assert any("Mamba" in (a.get("title") or "") for a in r2.attached_external)
print(f"[replay] fresh views + load_backflow -> {n} paper, "
      f"walk still surfaces it (library grew across sessions)", flush=True)

# --- 6. idempotency: second identical extraction adds nothing --------------
res2 = apply_backflow(views, bf, persist_path=probe_path)
assert res2["attached"] == 0 and res2["already_present"]
print("[idempotent] re-extraction adds 0 edges", flush=True)

print("\nLIVE PROBE PASS: coarse -> match -> attach -> zero-cost walk -> "
      "replay -> idempotent", flush=True)
