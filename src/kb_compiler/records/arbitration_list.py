# -*- coding: utf-8 -*-
"""Governance: pre-cluster a registry growth round's NEW entities into an
arbitration list for human review (never auto-applied — same discipline as
every schema/registry change).

Categories (heuristics are PROPOSALS; the human arbitrator has final say):
  junk_latex      : LaTeX debris / encoding garbage          -> drop
  ablation_variant: "w/o X", "1-step X", "A + B" combos      -> dims.variant, not registry
  hyperparam_cfg  : "CMR=0.5", "CPW(0.71)"                   -> config/dims.hyperparam
  group_reference : "C51, QR-DQN, IQN" (all parts already in registry) -> split, drop surface
  citation_ref    : "Asadi et al. [3]"                       -> drop
  generic_label   : "Basic", "Best baseline"                 -> drop
  benchmark_setup : benchmark/environment/domain surfaces    -> dims vocab, not registry
  merge_candidate : near-string match to an EXISTING registry canonical/alias -> merge?
  keep_new        : survives all flags                       -> confirm as new entity

Usage:
  python -m kb_compiler.records.arbitration_list \
      --growth registry_growth_report_v3.json --registry registry_v3.json \
      --out-md registry_v3_arbitration.md --out-json registry_v3_arbitration.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict

GENERIC_STOPLIST = {"basic", "best baseline", "baseline", "ours", "our method",
                    "the method", "method", "default", "standard", "none", "n/a"}
BENCH_NOTE_KEYS = ("benchmark", "environment", "domain", "task suite", "dataset")


def _flags(surface, canonical, note, existing_norm, vocab_norm):
    f = []
    s, c, n = surface or "", (canonical or "").lower(), (note or "").lower()
    if re.search(r"[\$\\]|(\^ ?\{)|mathbf|\?\?", s) or "^^" in c or "?" in c:
        f.append("junk_latex")
    if re.match(r"^(w/o|without)\b", c) or "ablation" in n or "variant" in n:
        f.append("ablation_variant")
    if re.search(r"=\s*-?\d|\(-?\d+(\.\d+)?\)$", c) or "hyperparameter" in n \
            or "configuration" in n:
        f.append("hyperparam_cfg")
    parts = [p.strip().lower() for p in re.split(r"[,;]| and ", c) if p.strip()]
    if len(parts) >= 2 and all(p in existing_norm for p in parts):
        f.append("group_reference")
    if re.search(r"et al\.?|\[\d+\]", s):
        f.append("citation_ref")
    if c in GENERIC_STOPLIST or "too generic" in n or "generic reference" in n:
        f.append("generic_label")
    if c in vocab_norm or any(k in n for k in BENCH_NOTE_KEYS):
        f.append("benchmark_setup")
    return f


def _merge_candidate(canonical, existing):
    if canonical in existing:
        return None
    m = difflib.get_close_matches(canonical, list(existing), n=1, cutoff=0.86)
    return m[0] if m else None


def build_list(growth, registry, vocab=None):
    new = growth.get("new", [])
    existing = {}   # norm name -> canonical
    for e in registry.get("entities", []):
        existing[e["canonical"].lower()] = e["canonical"]
        for a in e.get("aliases", []):
            existing[a.lower()] = e["canonical"]
    # entities added by THIS growth round are also merge targets for each other
    by_canon = defaultdict(list)
    for x in new:
        by_canon[(x.get("canonical") or x.get("surface") or "").lower()].append(x)
    vocab_norm = set()
    for fam in (vocab or {}).get("subject", []):
        vocab_norm.add(str(fam.get("family", "")).lower())
        vocab_norm.update(str(m).lower() for m in fam.get("members", []))
    for dim in ("setup", "variant", "hyperparam_items"):
        for e in (vocab or {}).get(dim, []):
            vocab_norm.add(str(e.get("canonical", "")).lower())
            vocab_norm.update(str(a).lower() for a in e.get("aliases", []))
            vocab_norm.update(str(a).lower() for a in e.get("family_hints", []))
    vocab_norm.discard("")

    rows, added = [], set()
    for canon, group in sorted(by_canon.items()):
        x0 = group[0]
        note = x0.get("note", "")
        flags = _flags(x0.get("surface"), canon, note, existing, vocab_norm)
        # a later round's own additions can absorb surfaces too
        target = _merge_candidate(canon, existing) or _merge_candidate(canon, added)
        if target and "junk_latex" not in flags:
            flags.append("merge_candidate")
        if not flags:
            flags = ["keep_new"]
        action = {
            "junk_latex": "DROP（LaTeX 残渣/乱码）",
            "citation_ref": "DROP（文献引用指称，非实体）",
            "generic_label": "DROP（泛指标签）",
            "group_reference": "SPLIT→已有实体（组合指称不单列）",
            "ablation_variant": "改道 dims.variant（消融变体非注册实体）",
            "hyperparam_cfg": "改道 config/dims.hyperparam（取值非实体）",
            "benchmark_setup": "改道 dims 词表 subject/setup（评测域非方法实体）",
            "merge_candidate": f"MERGE?→ {target}",
            "keep_new": "保留为新实体（请确认）",
        }[flags[0]]
        if "keep_new" == flags[0]:
            added.add(canon)
        rows.append({
            "canonical": canon, "surfaces": sorted({g["surface"] for g in group}),
            "entity_type": x0.get("entity_type"), "note": note,
            "flags": flags, "proposal": action,
        })
    order = ["junk_latex", "citation_ref", "generic_label", "group_reference",
             "ablation_variant", "hyperparam_cfg", "benchmark_setup",
             "merge_candidate", "keep_new"]
    rows.sort(key=lambda r: (order.index(r["flags"][0]), r["canonical"]))
    summary = defaultdict(int)
    for r in rows:
        summary[r["flags"][0]] += 1
    return {"n_new_surfaces": len(new), "n_canonical_groups": len(rows),
            "summary": {k: summary[k] for k in order if summary[k]}, "rows": rows}


def render_md(rep):
    L = [f"# Registry 生长轮新实体仲裁清单（{rep['n_new_surfaces']} 表面名 → "
         f"{rep['n_canonical_groups']} canonical 组）\n",
         "提案均为机器预聚类，**最终裁决在人**。每行动作：改判/确认请在行尾标注。\n",
         "## 汇总\n",
         "| 提案类别 | 组数 |", "|---|---|"]
    for k, v in rep["summary"].items():
        L.append(f"| {k} | {v} |")
    cur = None
    for r in rep["rows"]:
        cat = r["flags"][0]
        if cat != cur:
            cur = cat
            L.append(f"\n## {cat}\n")
            L += ["| canonical | surfaces | type | 提案 | note |", "|---|---|---|---|---|"]
        L.append(f"| {r['canonical']} | {'; '.join(r['surfaces'])} | "
                 f"{r['entity_type']} | {r['proposal']} | {r['note'][:80]} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    from .common import load_json, save_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--growth", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--vocab", default="")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    rep = build_list(load_json(args.growth, {}), load_json(args.registry, {}),
                     load_json(args.vocab, {}) if args.vocab else None)
    save_json(rep, args.out_json)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(render_md(rep))
    print(json.dumps(rep["summary"], ensure_ascii=False))
    print(f"-> {args.out_md}")
