# -*- coding: utf-8 -*-
"""Targeted chunk-failure top-up (reusable).

Steady-state reality: Paratera brownout floor ~5-8% of chunk calls fail even
with 3 retries + salvage (measured RL40 first pass 47/916, rerun 41/~500 —
different chunks each round = transient infrastructure, not content).
Whole-paper reruns have diminishing returns; this tool re-extracts ONLY the
failed chunks and merges into the records file (dedup by id, longest quote
wins — same merge semantics as slot.extract_paper, whose attach/resolve logic
is mirrored here).

Also audits the silent-loss point: the whole-text absence pass failing leaves
no CHUNK FAIL log line — papers with zero absence records are flagged for
optional absence-pass redo (--redo-absence).

Usage:
  python -m kb_compiler.records.chunk_retry --records REC.json --log SLOT.log \
      --texts DIR --manifest M.json --cards C.json --registry R.json \
      --vocab V.json --model Qwen3.8-Max [--redo-absence] [--dry]
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from .common import call_json, load_corpus, load_json, load_manifest, save_json
from .schema import SCHEMA_VERSION
from .slot import (ABSENCE_PROMPT, CHUNK_PROMPT, KIND_SLICES, QUOTE_FIRST_RULES,
                   _fingerprint, _rid, build_injection, chunk_text)

_lock = threading.Lock()
_FAIL_RE = re.compile(r"\[(\w+)#c(\d+)\] CHUNK FAIL")


def parse_failed(log_path: str) -> dict:
    failed = {}
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = _FAIL_RE.search(line)
        if m:
            failed.setdefault(m.group(1), set()).add(int(m.group(2)))
    return failed


def _attach(rec, pid, ch, registry):
    """mirror of slot.extract_paper record attach + canonical resolution."""
    rec["paper_id"] = pid
    rec["chunk_id"] = ch["chunk_id"]
    rec["section"] = ch["section"]
    rec["chunk_char_start"] = ch["char_start"]
    return rec


def retry(records_path, log_path, texts_dir, manifest, cards, registry, vocab,
          model, redo_absence=False, dry=False, workers=2):
    records = load_json(records_path, {})
    failed = parse_failed(log_path)
    texts = dict(load_corpus(texts_dir))
    man = load_manifest(manifest)
    cards = load_json(cards, {})
    registry = load_json(registry, {})
    vocab = load_json(vocab, {})
    n_chunks = sum(len(v) for v in failed.values())
    print(f"chunk_retry: {n_chunks} failed chunks across {len(failed)} papers"
          f"{' (dry)' if dry else ''}", flush=True)

    # absence-pass audit (silent failure point)
    no_absence = [pid for pid, p in records.items()
                  if not any(r.get("kind") == "absence" for r in p.get("records", []))]
    print(f"papers with zero absence records (suspect absence-pass failure): "
          f"{no_absence}", flush=True)

    if dry:
        for pid, cids in sorted(failed.items()):
            print(f"  {pid}: chunks {sorted(cids)}", flush=True)
        return

    si = registry.get("surface_index", {})
    byid = {e["entity_id"]: e for e in registry.get("entities", [])}
    stats = {"retried": 0, "recovered": 0, "still_failed": 0, "new_records": 0}

    def do_chunk(pid, cidx):
        text = texts.get(pid, "")
        if not text or pid not in cards:
            return pid, cidx, None
        chunks = chunk_text(pid, text)  # deterministic — same chunking as slot run
        if cidx >= len(chunks):
            return pid, cidx, None
        ch = chunks[cidx]
        inj = build_injection(pid, cards[pid], registry, vocab)
        title = (man.get(pid) or {}).get("title") or pid
        schemas = "\n".join(KIND_SLICES[k] for k in ch["kinds"] if k in KIND_SLICES)
        prompt = (CHUNK_PROMPT.replace("{title}", title)
                  .replace("{identity}", inj["identity"])
                  .replace("{schemas}", schemas)
                  .replace("{entities}", inj["entities"])
                  .replace("{subjects}", inj["subjects"])
                  .replace("{setups}", inj["setups"])
                  .replace("{variants}", inj["variants"])
                  .replace("{hparams}", inj["hparams"])
                  .replace("{rules}", QUOTE_FIRST_RULES)
                  .replace("{section}", ch["section"])
                  .replace("{chunk}", ch["text"]))
        obj = call_json(prompt, model, max_tokens=9000, retries=4, salvage=True)
        if isinstance(obj, list):
            obj = {"records": obj}
        if not isinstance(obj, dict):
            return pid, cidx, None  # failed again even with 4 retries + salvage
        return pid, cidx, (obj, ch)

    def do_absence(pid):
        text = texts.get(pid, "")
        card = cards.get(pid) or {}
        if not text or not card:
            return pid, None
        import json as _json
        inj = build_injection(pid, card, registry, vocab)
        matrix = _json.dumps(card.get("experimental_matrix") or [], ensure_ascii=False)[:3000]
        from .common import MAX_PAPER_CHARS
        prompt = (ABSENCE_PROMPT.replace("{title}", (man.get(pid) or {}).get("title") or pid)
                  .replace("{identity}", inj["identity"]).replace("{matrix}", matrix)
                  .replace("{schema}", KIND_SLICES["absence"])
                  .replace("{rules}", QUOTE_FIRST_RULES)
                  .replace("{text}", text[:MAX_PAPER_CHARS]))
        obj = call_json(prompt, model, max_tokens=6000, retries=4)
        return pid, obj

    # skip chunks already recovered by a previous (crashed) run — their records
    # carry chunk_id stamps; incremental save makes reruns idempotent
    jobs = []
    for pid, cids in sorted(failed.items()):
        have = {r.get("chunk_id") for r in (records.get(pid) or {}).get("records", [])}
        for c in sorted(cids):
            if f"{pid}#c{c}" in have:
                stats.setdefault("already_recovered", 0)
                stats["already_recovered"] += 1
                continue
            jobs.append((pid, c))
    print(f"jobs after skip-already-recovered: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do_chunk, pid, c) for pid, c in jobs]
        if redo_absence:
            futs += [ex.submit(do_absence, pid) for pid in no_absence]
        for f in futs:
            out = f.result()
            pid = out[0]
            if len(out) == 3:  # chunk job
                _, cidx, res = out
                stats["retried"] += 1
                if res is None:
                    stats["still_failed"] += 1
                    continue
                obj, ch = res
                stats["recovered"] += 1
                payload = records.setdefault(pid, {"records": [], "overflow": [],
                                                   "entity_queue": [], "stats": {}})
                existing = {r.get("id"): r for r in payload["records"]}
                for rec in obj.get("records") or []:
                    if not isinstance(rec, dict) or rec.get("kind") not in KIND_SLICES \
                            or rec["kind"] == "absence":
                        continue
                    _attach(rec, pid, ch, registry)
                    fp = _fingerprint(rec)
                    rec["id"] = _rid(pid, rec["kind"], fp)
                    for field in ("method", "from_method", "to_method", "scope_ref", "target_ref"):
                        surf = (rec.get(field) or "").strip() if isinstance(rec.get(field), str) else ""
                        if not surf:
                            continue
                        eid = si.get(re.sub(r"\s+", " ", surf.lower()))
                        if eid:
                            rec[field + "_ref"] = {"surface": surf,
                                                   "canonical": byid[eid]["canonical"],
                                                   "entity_id": eid}
                        else:
                            rec[field + "_ref"] = {"surface": surf, "canonical": None,
                                                   "entity_id": None}
                            payload.setdefault("entity_queue", []).append(
                                {"surface": surf, "paper_id": pid,
                                 "kind": rec["kind"], "field": field})
                        rec.pop(field, None)
                    prev = existing.get(rec["id"])
                    if prev is None:
                        payload["records"].append(rec)
                        existing[rec["id"]] = rec
                        stats["new_records"] += 1
                    elif len(rec.get("quote") or "") > len(prev.get("quote") or ""):
                        pos = payload["records"].index(prev)
                        payload["records"][pos] = rec
                        existing[rec["id"]] = rec  # BUGFIX(3rd): stale index entry
                        # caused ValueError on 3+ same-fingerprint records
                for ov in obj.get("overflow") or []:
                    if isinstance(ov, dict):
                        ov.update({"kind": "overflow", "paper_id": pid,
                                   "chunk_id": ch["chunk_id"], "section": ch["section"]})
                        payload.setdefault("overflow", []).append(ov)
                with _lock:
                    print(f"  [{pid}#c{cidx}] recovered ({len(obj.get('records') or [])} recs)",
                          flush=True)
                    save_json(records, records_path)  # incremental: crash loses ≤1 chunk
            else:  # absence job
                pid2, obj = out
                if obj and obj.get("records"):
                    payload = records.setdefault(pid2, {"records": [], "overflow": [],
                                                        "entity_queue": [], "stats": {}})
                    n = 0
                    for rec in obj["records"]:
                        if isinstance(rec, dict) and rec.get("kind") == "absence":
                            rec.update({"paper_id": pid2, "chunk_id": f"{pid2}#absence",
                                        "section": "(whole-text)", "chunk_char_start": 0})
                            rec["id"] = _rid(pid2, "absence", _fingerprint(rec))
                            payload["records"].append(rec)
                            n += 1
                    stats["new_records"] += n
                    print(f"  [{pid2}] absence pass redone: +{n}", flush=True)
                    with _lock:
                        save_json(records, records_path)

    # refresh per-paper stats
    for pid, p in records.items():
        p["stats"] = {"chunks": p.get("stats", {}).get("chunks"),
                      "records": len(p.get("records", [])),
                      "overflow": len(p.get("overflow", [])),
                      "queue": len(p.get("entity_queue", []))}
        p["schema_version"] = p.get("schema_version") or SCHEMA_VERSION
    save_json(records, records_path)
    print(json_stats := str(stats), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--model", default="Qwen3.8-Max")
    ap.add_argument("--redo-absence", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    retry(args.records, args.log, args.texts, args.manifest, args.cards,
          args.registry, args.vocab, args.model, args.redo_absence, args.dry,
          args.workers)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
