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
    # output truncation (3x parse fail, unassigned fallback flood); salvage_json_records
    # recovers truncated arrays, exactly its design purpose.
    obj = call_json(MAP_PROMPT.replace("{registry_lines}", registry_lines)
                    .replace("{queue_lines}", queue_lines),
                    model, max_tokens=12000, retries=3, salvage=True) or {}
    return obj.get("assignments") or []


def run_round2(registry: dict, queue: list[dict], model: str):
    canon_list = sorted({e["canonical"] for e in registry["entities"]})
    registry_lines = "\n".join(f"- {c}" for c in canon_list)
    # batched mapping (registry list is the shared join anchor across batches,
    # so batching does not reintroduce the slice-isolation fragmentation)
    assigns = []
    for b0 in range(0, len(queue), BATCH_SIZE):
        batch = queue[b0:b0 + BATCH_SIZE]
        got = _map_batch(batch, registry_lines, model)
        by_i_batch = {}
        for a in got:
            if isinstance(a, dict) and isinstance(a.get("i"), int):
                by_i_batch.setdefault(a["i"], a)
        for i in range(len(batch)):
            a = dict(by_i_batch.get(i) or {"action": "new", "canonical": batch[i]["surface"],
                                           "entity_type": "method", "note": "batch_unassigned"})
            a["_qidx"] = b0 + i
            assigns.append(a)
        print(f"  batch {b0//BATCH_SIZE + 1}: {len(batch)} surfaces, "
              f"{sum(1 for i in range(len(batch)) if i in by_i_batch)} assigned", flush=True)
    # consolidation pass over proposed NEW entities (dedupe cross-batch twins)
    new_props = {}
    for a in assigns:
        if a.get("action") != "match":
            new_props.setdefault(re.sub(r"\s+", " ", str(a.get("canonical") or "").strip().lower()),
                                 []).append(a["_qidx"])
    if len(new_props) > 25:
        prop_lines = "\n".join(f"[{i}] {c}" for i, c in enumerate(sorted(new_props)))
        cobj = call_json(
            "下面是新实体提案清单（编号|规范名）。合并指向同一实体的提案（缩写/变体/大小写）。"
            "每个编号恰好出现一次。输出 JSON：{\"groups\": [{\"canonical\": 编号, \"members\": [编号,...]}]}\n\n"
            + prop_lines, model, max_tokens=8000, retries=2) or {}
        prop_keys = sorted(new_props)
        groups = (cobj.get("groups") or [])
        seen_idx = set()
        rename = {}
        for g in groups:
            mem = [m for m in (g.get("members") or []) if isinstance(m, int)
                   and 0 <= m < len(prop_keys) and m not in seen_idx]
            if not mem:
                continue
            seen_idx.update(mem)
            can = g.get("canonical") if isinstance(g.get("canonical"), int) and g["canonical"] in mem else mem[0]
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
            target = next(e for e in registry["entities"] if e["canonical"] == a["canonical"])
            if s["surface"] not in target["aliases"]:
                target["aliases"].append(s["surface"])
            index_additions[re.sub(r"\s+", " ", s["surface"].strip().lower())] = target["entity_id"]
        else:
            etype = a.get("entity_type") if a.get("entity_type") in (
                "method", "mechanism", "practice", "out_of_corpus") else "method"
            import hashlib
            canonical = (a.get("canonical") or s["surface"]).strip()
            ent = {"entity_id": hashlib.md5(canonical.lower().encode()).hexdigest()[:12],
                   "canonical": canonical, "aliases": sorted({canonical, s["surface"]}),
                   "entity_type": etype, "in_corpus_paper_id": None,
                   "origin_year_cited": None, "mention_papers": s["papers"],
                   "mention_count": len(s["papers"]),
                   "provenance": "round2_growth"}
            new_entities.append(ent)
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
    args = ap.parse_args()

    registry = load_json(args.registry, {})
    queue = collect_queue(args.records)
    print(f"round2: {len(queue)} unique unresolved surfaces from "
          f"{len(args.records)} record file(s)", flush=True)
    if not queue:
        print("nothing to fold; registry unchanged", flush=True)
        return
    registry2, report = run_round2(registry, queue, args.model)
    v = registry2["version"]
    save_json(registry2, f"{args.out_dir}/registry_v{v}.json")
    save_json(report, f"{args.out_dir}/registry_growth_report_v{v}.json")
    print(f"registry v{v}: matched={len(report['matched'])} "
          f"new={len(report['new'])} fallback={len(report['unassigned_fallback'])} "
          f"| total entities={len(registry2['entities'])}", flush=True)
    print("NEW entities are the arbitration queue (human review before vN+1 "
          "becomes the run-declared registry)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
