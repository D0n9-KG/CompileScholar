# -*- coding: utf-8 -*-
"""GOLDCOV step 1: render the 18,989 records (53 papers, v39) into a
searchable record index.

Reuses the A7 TextSearchIndex on-disk format exactly (chunks.jsonl + emb.bin
+ meta.json, provider cst-qwen3) so step-3 retrieval is a plain TextSearchIndex
load. One chunk == one record (no re-chunking: records are atomic units and
must be retrieved whole).

paper_id field of each chunk = the RECORD id (traceability); the true paper
id and kind live in rec_map.json (record_id -> {paper_id, kind, section}).

Usage:
  py -3.13 render_records.py            # build index + rec_map
  py -3.13 render_records.py --probe "NEFTune AlpacaEval win rate"
"""
import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = os.path.dirname(HERE)
SRC = "C:/Users/D0n9/Desktop/CompileScholar/src"
sys.path.insert(0, SRC)

RECORDS = os.path.join(D2, "records_final_d2_v39.json")
OUT_DIR = os.path.join(HERE, "rec_index")

SKIP_FIELDS = {"id", "chunk_id", "chunk_char_start", "loc", "provenance",
               "repair_history", "dims_new", "dims"}
REF_FIELDS = {"method_ref", "from_method_ref", "to_method_ref", "target_ref",
              "scope_ref", "scope_ref_ref", "target_ref_ref"}


def _val(v, depth=0):
    """Render a field value compactly (str/dict/list)."""
    if v is None or v == "" or v == {} or v == []:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if depth > 1:
            return json.dumps(v, ensure_ascii=False)[:120]
        parts = []
        for k2, v2 in v.items():
            r = _val(v2, depth + 1)
            if r is not None:
                parts.append(f"{k2}={r}")
        return "; ".join(parts) if parts else None
    if isinstance(v, list):
        parts = [x for x in (_val(x, depth + 1) for x in v) if x is not None]
        return "; ".join(parts)[:300] if parts else None
    return str(v)


def render_record(r):
    """One record -> one searchable text line."""
    parts = [f"[{r.get('kind')}]"]
    for k, v in r.items():
        if k in SKIP_FIELDS or k == "kind":
            continue
        if k in REF_FIELDS and isinstance(v, dict):
            surf = v.get("canonical") or v.get("surface")
            if surf:
                parts.append(f"{k}={surf}")
            continue
        val = _val(v)
        if val is not None:
            parts.append(f"{k}={val}")
    parts.append(f"paper={r.get('paper_id')}")
    text = " | ".join(parts)
    return text[:1500]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    recs = json.load(open(RECORDS, encoding="utf-8"))
    os.makedirs(OUT_DIR, exist_ok=True)
    chunks, rmap = [], {}
    for pid, v in recs.items():
        for r in v["records"]:
            rid = r["id"]
            text = render_record(r)
            if len(text.strip()) < 20:
                continue
            chunks.append({"paper_id": rid, "char_start": 0, "char_end": len(text),
                          "text": text})
            rmap[rid] = {"paper_id": pid, "kind": r.get("kind"),
                         "section": r.get("section", "")[:80]}
    print(f"rendered {len(chunks)} records", flush=True)
    # embed (same provider discipline: cst-qwen3 4096-dim for corpus+query)
    from array import array
    from kb_infra.embedding import embed_cst
    print(f"embedding {len(chunks)} record texts...", flush=True)
    embs = embed_cst([c["text"] for c in chunks])
    with open(os.path.join(OUT_DIR, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "emb.bin"), "wb") as f:
        array("f", (x for e in embs for x in e)).tofile(f)
    json.dump({"n_chunks": len(chunks), "dim": len(embs[0]),
               "provider": "cst-qwen3", "chunk_chars": 1500, "overlap": 0},
              open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8"))
    json.dump(rmap, open(os.path.join(HERE, "rec_map.json"), "w",
                         encoding="utf-8"))
    print(f"index built -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe")
    a = ap.parse_args()
    if a.probe:
        sys.path.insert(0, os.path.join(SRC, "kb_compiler", "views"))
        from search_text import TextSearchIndex
        idx = TextSearchIndex(OUT_DIR)
        rmap = json.load(open(os.path.join(HERE, "rec_map.json"), encoding="utf-8"))
        res = idx.search(a.probe, k=8)
        for h in res["hits"]:
            m = rmap.get(h["paper_id"], {})
            print(f"[{h['score']}] {m.get('paper_id')} {m.get('kind')} :: {h['text'][:150]}")
    else:
        main()
