# -*- coding: utf-8 -*-
"""Registry growth round 2: fold Stage-2 entity_queue surfaces into the
registry BETWEEN runs (spec v1.1 §2 growth discipline: the registry grows at
registration rounds, never at extraction time; versioned, arbitration-visible).

Input : registry vN + entity_queue entries collected from slot outputs
Output: registry vN+1 (new file; vN untouched) + growth report
        (merged-into-existing vs new entities — new entities are the
        arbitration queue for human review)

Mechanism: one LLM call maps each queued surface onto an existing entity or
proposes a new one (index-based output + machine coverage check, same
anti-truncation design as round 1).

Usage:
  python -m kb_compiler.records.registry_round2 --registry R.json \
      --records rec1.json [rec2.json ...] --out-dir DIR --model DeepSeek-V4-Flash
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

from .common import call_json, load_json, save_json

MAP_PROMPT = """下面是抽取管线产出的未解析实体表面名（编号|表面名|出处上下文），以及现有实体注册表的规范名清单。

任务：为每个编号的表面名判定：
- match: 与某个现有实体相同 → 给出该实体的规范名（必须逐字从清单选）
- new: 是新实体 → 给出 canonical 名（选最通用的表面写法）+ entity_type（method|mechanism|practice|out_of_corpus）

规则：只在确信相同时 match；同形不同义（同名不同领域实体）标 new 并在 note 说明；每个编号恰好出现一次。

输出 JSON：{{"assignments": [{{"i": 编号, "action": "match|new", "canonical": "...", "entity_type": "...(仅new)", "note": ""}}]}}
只输出 JSON。

现有实体规范名清单：
{registry_lines}

未解析表面名：
{queue_lines}"""



# raw outputs that failed direct JSON parse (diagnosis only); repo-relative (10-04: was an absolute user path)
_FAIL_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".research_tmp", "experiments",
                          "benchmarks", "scholarqa_multi", "kb", "forensic_map_fail.txt")

def _norm(s: str) -> str:
    # keep in sync with registry._norm (entity ids are md5 of this)
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def collect_queue(record_files: list[str]) -> list[dict]:
    q, seen = [], {}
    for path in record_files:
        data = load_json(path, {}) or {}
        for pid, payload in data.items():
            if pid == "canary":
                continue  # synthetic sentinel entities must never enter the
                # production registry (v2 first run caught this contamination)
            for e in payload.get("entity_queue", []):
                key = re.sub(r"\s+", " ", (e.get("surface") or "").strip().lower())
                if not key:
                    continue
                s = seen.setdefault(key, {"surface": e["surface"], "papers": set(),
                                          "contexts": []})
                s["papers"].add(e.get("paper_id"))
                if len(s["contexts"]) < 3:
                    s["contexts"].append(f"{e.get('kind')}.{e.get('field')}")
    for key, s in seen.items():
        s["papers"] = sorted(s["papers"])
        q.append(s)
    return sorted(q, key=lambda s: (-len(s["papers"]), s["surface"]))


BATCH_SIZE = 80   # single-call truncation guard: 500+ surfaces x per-item JSON
                  # would blow max_tokens -> coverage fallback floods the
                  # arbitration list with fake singletons (measured risk 2026-09-06;
                  # IL-P1: 120 still truncated on PS16 batch-1, lowered to 80)


def _map_batch(batch: list[dict], registry_lines: str, model: str) -> list:
    queue_lines = "\n".join(
        f"[{i}] {s['surface']} | papers={','.join(s['papers'][:3])} | "
        f"ctx={','.join(s['contexts'][:2])}" for i, s in enumerate(batch))
    # IL-P1 (pilot): salvage tier ON — batch-1 of PS16 run lost 120 surfaces to
    # output truncation (3x parse fail, unassigned fallback flood); salvage
    # recovers the complete prefix. 2026-09-23 run5 forensics: salvage was
    # silently empty for assignment-shaped output (only matched "kind") —
    # now key/wrapper-generalized. fail_dump captures the FULL raw of every
    # unparseable response (even when salvage recovers) — the previous debug
    # capture capped at 2000 chars showed a clean head and hid the tail.
    from .registry import _must_json
    obj = _must_json(call_json(
        MAP_PROMPT.replace("{registry_lines}", registry_lines)
        .replace("{queue_lines}", queue_lines),
        model, max_tokens=12000, retries=3, salvage=True,
        salvage_key="i", salvage_wrapper="assignments",
        fail_dump=_FAIL_DUMP),
        "round2 mapping batch")
    return obj.get("assignments") or []


UNION_CAP = 600   # per-chunk candidate anchor bound (prompt-size); scale mode
                  # replaces the full registry anchor with top-k nearest
                  # canonicals per surface (Multi-431: ~15k canonicals do not
                  # fit any context window — KB-SCALING-DESIGN.md §2 round2)
TOP_K = 12


def _build_chunks(queue, canon_list, embed_cache_dir=None,
                  top_k=TOP_K, union_cap=UNION_CAP):
    """[(query_indices, registry_lines)] — legacy full-anchor batches when the
    registry is small; embedding top-k candidate anchors when it is not."""
    if len(canon_list) <= union_cap:
        full = "\n".join(f"- {c}" for c in canon_list)
        return [(list(range(b0, min(b0 + BATCH_SIZE, len(queue)))), full)
                for b0 in range(0, len(queue), BATCH_SIZE)], "full-anchor"
    from .embed_block import (embed_items, topk_candidates,
                              chunk_by_candidate_union)
    cemb = embed_items(canon_list,
                       os.path.join(embed_cache_dir, "embed_r2_canon.json"))
    qemb = embed_items([s["surface"] for s in queue],
                       os.path.join(embed_cache_dir, "embed_r2_queue.json"))
    cand = topk_candidates(qemb, cemb, top_k=top_k)
    raw_chunks = chunk_by_candidate_union(cand, chunk_size=BATCH_SIZE,
                                          union_cap=union_cap)
    chunks = []
    for ch in raw_chunks:
        union = {ci for qi in ch for ci in cand[qi]}
        if len(union) > union_cap:
            # trim by candidate rank (best matches survive), deterministic
            kept = []
            seen = set()
            for rank in range(top_k):
                for qi in ch:
                    if rank < len(cand[qi]) and cand[qi][rank] not in seen:
                        seen.add(cand[qi][rank])
                        kept.append(cand[qi][rank])
                    if len(seen) >= union_cap:
                        break
                if len(seen) >= union_cap:
                    break
            union = set(kept)
        lines = "\n".join(f"- {canon_list[ci]}" for ci in sorted(union))
        chunks.append((ch, lines))
    return chunks, "topk-anchor"


def run_round2(registry: dict, queue: list[dict], model: str,
               embed_cache_dir: str | None = None,
               checkpoint_dir: str | None = None):
    canon_list = sorted({e["canonical"] for e in registry["entities"]})
    ent_by_canon: dict[str, dict] = {}
    for e in registry["entities"]:
        ent_by_canon.setdefault(e["canonical"], e)
    chunks, anchor_mode = _build_chunks(queue, canon_list,
                                        embed_cache_dir=embed_cache_dir)
    print(f"round2 anchors: {anchor_mode} ({len(canon_list)} canonicals, "
          f"{len(chunks)} chunks)", flush=True)
    # batched mapping (small registry: full list is the shared join anchor,
    # so batching does not reintroduce slice-isolation fragmentation; large
    # registry: per-chunk top-k candidate anchors — a true canonical missing
    # from the top-k lands in `new` and is caught by the consolidation pass
    # and the human arbitration queue, never silently dropped)
    from .common import par_map

    # chunk WAL (2026-09-23): this stage died 5x overnight; each death lost
    # ALL mapping calls (~4800 completion tokens x surviving chunks). Chunks
    # are deterministic from (registry, records) inputs — validated against
    # first surface + size before reuse, stale rows ignored.
    wal_path = None
    wal_done: dict[int, list] = {}
    _wal_append = None
    if checkpoint_dir:
        import threading
        os.makedirs(checkpoint_dir, exist_ok=True)
        wal_path = os.path.join(checkpoint_dir, "registry_growth_round2.wal.jsonl")
        _wal_lock = threading.Lock()

        def _wal_append(bn: int, n: int, first: str, out: list):
            with _wal_lock:
                with open(wal_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"bn": bn, "n": n, "first": first,
                                        "out": out}, ensure_ascii=False) + "\n")

        if os.path.exists(wal_path):
            with open(wal_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    bn = rec.get("bn")
                    if (isinstance(bn, int) and 0 <= bn < len(chunks)
                            and rec.get("n") == len(chunks[bn][0])
                            and rec.get("first") == queue[chunks[bn][0][0]]["surface"]):
                        wal_done[bn] = rec["out"]
            if wal_done:
                print(f"round2 WAL: resuming, {len(wal_done)}/{len(chunks)} "
                      f"chunks already mapped", flush=True)

    def _run_chunk(item):
        bn, (ch, registry_lines) = item
        if bn in wal_done:
            out = wal_done[bn]
            return bn, len(out), -1, out
        batch = [queue[i] for i in ch]
        got = _map_batch(batch, registry_lines, model)
        by_i_batch = {}
        for a in got:
            if isinstance(a, dict) and isinstance(a.get("i"), int):
                by_i_batch.setdefault(a["i"], a)
        out = []
        for i in range(len(batch)):
            a = dict(by_i_batch.get(i) or {"action": "new", "canonical": batch[i]["surface"],
                                           "entity_type": "method", "note": "batch_unassigned"})
            a["_qidx"] = ch[i]
            out.append(a)
        n_assigned = sum(1 for i in range(len(batch)) if i in by_i_batch)
        if _wal_append:
            _wal_append(bn, len(batch), batch[0]["surface"], out)
        return bn, len(batch), n_assigned, out

    assigns = []
    results = par_map(_run_chunk, list(enumerate(chunks)))
    for bn, nb, n_assigned, out in results:
        assigns.extend(out)
        if (bn + 1) % 10 == 0 or bn == len(chunks) - 1:
            print(f"  batch {bn + 1}/{len(chunks)}: {nb} surfaces, "
                  f"{n_assigned} assigned", flush=True)
    # consolidation pass over proposed NEW entities (dedupe cross-batch twins)
    new_props = {}
    for a in assigns:
        if a.get("action") != "match":
            new_props.setdefault(re.sub(r"\s+", " ", str(a.get("canonical") or "").strip().lower()),
                                 []).append(a["_qidx"])
    if len(new_props) > 25:
        prop_keys = sorted(new_props)
        _CONSOL_PROMPT = (
            "下面是新实体提案清单（编号|规范名）。合并指向同一实体的提案（缩写/变体/大小写）。"
            "每个编号恰好出现一次。输出 JSON：{\"groups\": [{\"canonical\": 编号, \"members\": [编号,...]}]}\n\n")
        # scale mode: one call over thousands of proposals truncates (8k cap);
        # embed-blocked consolidation, same conservative-block design as round1
        if len(prop_keys) > 400 and embed_cache_dir:
            from .embed_block import embed_items, block_indices
            pembs = embed_items(prop_keys,
                                os.path.join(embed_cache_dir, "embed_r2_props.json"))
            pblocks = block_indices(pembs, prop_keys, tau=0.80, max_block=300)
        else:
            pblocks = [list(range(len(prop_keys)))]
        groups = []
        # consolidation block pool (2026-09-22): blocks are independent LLM
        # calls; the serial loop is the last sequential stage in this module
        from concurrent.futures import ThreadPoolExecutor

        def _consol_one(pb):
            if len(pb) == 1:
                return []
            prop_lines = "\n".join(f"[{pos}] {prop_keys[gi]}"
                                   for pos, gi in enumerate(pb))
            from .registry import _must_json
            cobj = _must_json(call_json(
                _CONSOL_PROMPT + prop_lines, model,
                max_tokens=8000, retries=2, salvage=True,
                salvage_key="members", salvage_wrapper="groups",
                fail_dump=os.path.join(os.path.dirname(_FAIL_DUMP), "forensic_consol_fail.txt")),
                "round2 proposal consolidation")
            return [(g, pb) for g in (cobj.get("groups") or [])]

        with ThreadPoolExecutor(max_workers=12) as cex:
            for gs in cex.map(_consol_one, pblocks):
                for g, pb in gs:
                    groups.append((g, pb))
        # flatten pooled results (original block order preserved by cex.map)
        flat = []
        for g, pb in groups:
            mem = [pb[m] for m in (g.get("members") or [])
                   if isinstance(m, int) and 0 <= m < len(pb)]
            if not mem:
                continue
            can = pb[g["canonical"]] if isinstance(g.get("canonical"), int)                 and g["canonical"] in range(len(pb)) else mem[0]
            flat.append({"canonical": can, "members": mem})
        groups = flat
        seen_idx = set()
        rename = {}
        for g in groups:
            mem = [m for m in g["members"] if m not in seen_idx]
            if not mem:
                continue
            seen_idx.update(mem)
            can = g["canonical"] if g["canonical"] in mem else mem[0]
            target = prop_keys[can]
            for m in mem:
                rename[prop_keys[m]] = target
        for a in assigns:
            if a.get("action") != "match":
                key = re.sub(r"\s+", " ", str(a.get("canonical") or "").strip().lower())
                if key in rename:
                    a["canonical"] = rename[key]
    by_i = {}
    for a in assigns:
        by_i.setdefault(a["_qidx"], a)
    queue_len = len(queue)
    # coverage: unassigned surfaces -> new entity with surface as canonical
    report = {"matched": [], "new": [], "unassigned_fallback": []}
    canon_set = set(canon_list)
    # G3 dup guard (2026-09-24): action=new with a canonical that already
    # exists (case/space variant) used to md5-collide with the existing
    # entity_id and append a duplicate row — v2 carried 1,450 such rows that
    # views/cards silently overwrote. Fold those into the existing entity.
    ent_by_norm: dict[str, dict] = {}
    for e in registry["entities"]:
        ent_by_norm.setdefault(_norm(e["canonical"]), e)
    new_by_norm: dict[str, dict] = {}
    new_entities = []
    index_additions = {}
    for i, s in enumerate(queue):
        a = by_i.get(i)
        if a is None:
            report["unassigned_fallback"].append(s["surface"])
            a = {"action": "new", "canonical": s["surface"], "entity_type": "method"}
        if a.get("action") == "match" and a.get("canonical") in canon_set:
            report["matched"].append({"surface": s["surface"],
                                      "canonical": a["canonical"]})
            target = ent_by_canon[a["canonical"]]
            if s["surface"] not in target["aliases"]:
                target["aliases"].append(s["surface"])
            index_additions[re.sub(r"\s+", " ", s["surface"].strip().lower())] = target["entity_id"]
        else:
            etype = a.get("entity_type") if a.get("entity_type") in (
                "method", "mechanism", "practice", "out_of_corpus") else "method"
            import hashlib
            canonical = (a.get("canonical") or s["surface"]).strip()
            target = ent_by_norm.get(_norm(canonical)) or new_by_norm.get(_norm(canonical))
            if target is not None:
                # LLM proposed a canonical variant of an already-registered
                # entity — fold as alias, never append a duplicate row
                report.setdefault("dup_guard_folds", []).append(
                    {"surface": s["surface"], "canonical": target["canonical"]})
                if s["surface"] not in target["aliases"]:
                    target["aliases"].append(s["surface"])
                index_additions[re.sub(r"\s+", " ", s["surface"].strip().lower())] = target["entity_id"]
                continue
            ent = {"entity_id": hashlib.md5(_norm(canonical).encode()).hexdigest()[:12],
                   "canonical": canonical, "aliases": sorted({canonical, s["surface"]}),
                   "entity_type": etype, "in_corpus_paper_id": None,
                   "origin_year_cited": None, "mention_papers": s["papers"],
                   "mention_count": len(s["papers"]),
                   "provenance": "round2_growth"}
            new_entities.append(ent)
            new_by_norm[_norm(canonical)] = ent
            report["new"].append({"surface": s["surface"], "canonical": canonical,
                                  "entity_type": etype, "note": a.get("note", "")})
            index_additions[re.sub(r"\s+", " ", s["surface"].strip().lower())] = ent["entity_id"]
    registry["entities"] += new_entities
    registry["surface_index"].update(index_additions)
    _v = registry.get("version")
    if isinstance(_v, str):
        m = re.match(r"^(\d+)\.(\d+)", _v)
        registry["version"] = (f"{m.group(1)}.{int(m.group(2)) + 1}" if m
                               else _v + "-r2")
    else:
        registry["version"] = (_v or 1) + 1
    return registry, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--embed-cache", default="",
                    help="dir for embedding caches (default: OUT_DIR/embed_cache); "
                         "used only when the registry anchor exceeds UNION_CAP")
    args = ap.parse_args()

    registry = load_json(args.registry, {})
    queue = collect_queue(args.records)
    print(f"round2: {len(queue)} unique unresolved surfaces from "
          f"{len(args.records)} record file(s)", flush=True)
    if not queue:
        print("nothing to fold; registry unchanged", flush=True)
        return
    cache = args.embed_cache or os.path.join(args.out_dir, "embed_cache")
    os.makedirs(cache, exist_ok=True)
    registry2, report = run_round2(registry, queue, args.model,
                                   embed_cache_dir=cache,
                                   checkpoint_dir=args.out_dir)
    v = registry2["version"]
    save_json(registry2, f"{args.out_dir}/registry_v{v}.json")
    save_json(report, f"{args.out_dir}/registry_growth_report_v{v}.json")
    wal = os.path.join(args.out_dir, "registry_growth_round2.wal.jsonl")
    if os.path.exists(wal):
        os.remove(wal)
    print(f"registry v{v}: matched={len(report['matched'])} "
          f"new={len(report['new'])} fallback={len(report['unassigned_fallback'])} "
          f"| total entities={len(registry2['entities'])}", flush=True)
    print("NEW entities are the arbitration queue (human review before vN+1 "
          "becomes the run-declared registry)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
