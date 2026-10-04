# -*- coding: utf-8 -*-
"""Answer-layer adapter (prereg §3.1): KB paper_id -> per-question ctx index.

Our answer stack cites [paper_id] where paper_id = texts/ stem (KB pid).
Official Citation F1 expects [0],[1],... zero-based indices into each
question's ctxs list. This tool builds the deterministic bridge:

    stem <- manifest.corpus_pid <- m-hash <- corpus_map[title].paper_id
         <- question ctx title (exact string, corpus_map is keyed by title)

Output: kb/id_mapping.json  {qid: {stem: [ctx_idx, ...], ...}, ...}
(a stem can serve MULTIPLE ctx indices — two question-ctx entries share one
file in the Label-free/Laser-cooling pairs; the official hook semantics for
1->many are verified at S2 wiring time, we keep the full list.)

Also reports unmapped ctxs (papers with no text — the 4 unreachable + any
title mismatch) for the disclosure list and for the id_mapping fault-rate
red line (prereg §6: adapter failure > 5% halts the run).

Usage: python build_id_mapping.py
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "corpus")
KB = os.path.join(BASE, "kb")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    items = json.load(open(os.path.join(BASE, "data", "scholarqa_multi.json"),
                           encoding="utf-8"))
    cmap = json.load(open(os.path.join(CORPUS, "corpus_map.json"), encoding="utf-8"))
    manifest = json.load(open(os.path.join(CORPUS, "manifest.json"), encoding="utf-8"))

    # m-hash -> stem(s)
    hash2stems = {}
    for row in manifest:
        for h in row["corpus_pid"]:
            hash2stems.setdefault(h, []).append(row["paper_id"])

    mapping = {}
    total_ctx = mapped_ctx = 0
    unmapped = []          # (qid, ctx_idx, title, reason)
    title_miss = set()
    for q in items:
        qid = str(q.get("id"))
        qmap = {}
        for i, c in enumerate(q.get("ctxs") or []):
            total_ctx += 1
            t = (c.get("title") or "").strip()
            entry = cmap.get(t)
            if entry is None:
                unmapped.append((qid, i, t, "title not in corpus_map"))
                title_miss.add(t)
                continue
            h = entry.get("paper_id")
            stems = hash2stems.get(h)
            if not stems:
                unmapped.append((qid, i, t, f"m-hash {h} has no text (unreachable)"))
                continue
            for st in stems:
                qmap.setdefault(st, [])
                if i not in qmap[st]:
                    qmap[st].append(i)
            mapped_ctx += 1
        mapping[qid] = qmap

    os.makedirs(KB, exist_ok=True)
    out = os.path.join(KB, "id_mapping.json")
    json.dump(mapping, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_q_full = sum(1 for q in items
                   if not any(u[0] == str(q.get("id")) for u in unmapped))
    print(f"id_mapping: {len(mapping)} questions -> {out}")
    print(f"ctx coverage: {mapped_ctx}/{total_ctx} ({mapped_ctx/total_ctx:.4f}) "
          f"| adapter fault rate {1 - mapped_ctx/total_ctx:.4f} (red line 0.05)")
    print(f"questions with ALL ctxs mapped: {n_q_full}/{len(items)}")
    print(f"unmapped ctxs: {len(unmapped)}")
    for qid, i, t, why in unmapped:
        print(f"  q{qid} ctx[{i}]: {t[:60]} — {why}")
    shared = [(qid, st, idxs) for qid, qm in mapping.items()
              for st, idxs in qm.items() if len(idxs) > 1]
    print(f"stems serving multiple ctx indices (1->many): {len(shared)}")
    for qid, st, idxs in shared[:10]:
        print(f"  q{qid}: {st[:55]} -> {idxs}")


if __name__ == "__main__":
    main()
