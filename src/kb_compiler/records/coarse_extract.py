# -*- coding: utf-8 -*-
"""Tier-1 coarse extractor: abstract -> lightweight structured records.

Design (user directive 2026-09-25): retrieval cannot afford the full
extraction pipeline (cards->deep_extract->postcheck, ~2min/paper even at
pool concurrency) for every retrieved candidate, and offline pre-extraction
of everything users might need is impossible. Investment ladder:

  Tier 0  metadata (title+abstract, free from retrieval results)
  Tier 1  THIS MODULE: one light LLM call per abstract -> coarse records
          in the SAME record schema subset (kind/subject/claim/quote), so
          they flow into the same views pipeline and typed tools, marked
          provenance="coarse" (no full-text quote anchoring; the quote
          field carries the abstract sentence instead)
  Tier 2  full pipeline (existing), triggered selectively by the answer
          loop when a paper is judged core evidence

Coarse records support: relevance judgment (is this paper worth Tier 2?),
gap/lineage signal detection (does this abstract claim to fill a recorded
gap / extend a known method?), and early answer drafting. They are NOT
citation-grade evidence — postcheck gates skip them (no full text to
anchor against), and the answering prompt marks them as abstract-level.

Abstract-adapted contract (user directive: 复用 schema 子集，但适配摘要):
  - kinds: finding / method / limitation  (config/result need full text)
  - quote = the abstract sentence the claim is grounded in (verbatim)
  - epistemic defaults to "stated" (abstract claims are unverified by
    definition — the paper's own words about its results)
  - ids: "coarse:" prefix so they can never collide with Tier-2 record ids
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "src"))

from kb_compiler.records.common import call_json, route_model  # noqa: E402
from kb_infra.llm import parse_json_response  # noqa: E402

COARSE_PROMPT = """You extract structured records from ONE paper abstract. This is a coarse, abstract-only pass — no full text is available.

Abstract of "{title}" ({year}):

{abstract}

Extract 3-8 records. Output strict JSON:
{{"records": [
 {{"kind": "finding|method|limitation",
   "subject": "<main method/system/material name this record is about>",
   "claim": "<one-sentence factual claim from the abstract, faithful wording>",
   "claim_type": "<for findings: mechanism|observation|recommendation|criticism|definition|qualitative_ablation>",
   "quote": "<the abstract sentence the claim comes from, copied verbatim>",
   "mentions": ["<other named methods/systems/datasets this abstract mentions>"]}
 ]}}

Rules:
- claim must be grounded in the abstract verbatim wording — no external knowledge, no inference beyond the text
- "method" records: what method/system this paper proposes or uses (subject = its name)
- "limitation" records: any stated limitation, unresolved issue, or future-work gap
- "mentions": names of OTHER methods/systems the abstract references (lineage hooks — these link the paper to existing method families)
- If the abstract is too thin to extract anything reliable, return {{"records": []}}
- Output ONLY the JSON."""


def extract_coarse(title: str, abstract: str, year, model: str) -> dict:
    """One abstract -> coarse record payload. Returns the paper-keyed
    payload in the same shape as records_slot.json entries (minus the
    full-text bookkeeping): {records: [...], provenance: "coarse"}."""
    abstract = (abstract or "").strip()
    if len(abstract) < 150:
        return {"records": [], "provenance": "coarse",
                "note": "abstract too short for extraction"}
    prompt = COARSE_PROMPT.replace("{title}", (title or "untitled")[:200]) \
        .replace("{year}", str(year or "n/a")) \
        .replace("{abstract}", abstract[:6000])
    obj = call_json(prompt, model, max_tokens=2000, retries=2,
                    salvage=True, salvage_key="records")
    if not isinstance(obj, dict):
        return {"records": [], "provenance": "coarse",
                "note": "parse failed"}
    out = []
    for r in (obj.get("records") or [])[:10]:
        if not isinstance(r, dict) or not (r.get("claim") or "").strip():
            continue
        kind = r.get("kind") if r.get("kind") in (
            "finding", "method", "limitation") else "finding"
        out.append({
            "id": "coarse:" + _stable_id(title, r.get("claim")),
            "kind": kind,
            "subject": (r.get("subject") or "").strip()[:120],
            "claim": (r.get("claim") or "").strip()[:600],
            "claim_type": r.get("claim_type") if kind == "finding" else None,
            "quote": (r.get("quote") or "")[:400],
            "epistemic": "stated",
            "mentions": [str(m)[:80] for m in (r.get("mentions") or [])[:8]],
            "provenance": "coarse",
        })
    return {"records": out, "provenance": "coarse"}


def _stable_id(title: str, claim: str) -> str:
    import hashlib
    blob = f"{(title or '')[:120]}|{(claim or '')[:120]}"
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]


# ---------------- CLI (batch over a retrieval-result JSON) ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="JSON: list of {title, abstract, year, doi?} "
                         "or {paper_id: {...}} dict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="local:Qwen3.8-27B")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = json.load(open(args.input, encoding="utf-8"))
    if isinstance(data, dict):
        items = [(k, v) for k, v in data.items()]
    else:
        items = [(v.get("doi") or v.get("title", f"item{i}")[:40], v)
                 for i, v in enumerate(data)]
    items = [(k, v) for k, v in items
             if (v.get("abstract") or "").strip()]
    print(f"[coarse] {len(items)} papers with abstracts", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_coarse, v.get("title"), v.get("abstract"),
                          v.get("year"), args.model): k
                for k, v in items}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                results[k] = fut.result()
            except Exception as e:
                results[k] = {"records": [], "provenance": "coarse",
                              "note": f"error: {e}"}
    n_recs = sum(len(v.get("records", [])) for v in results.values())
    json.dump(results, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[coarse] {n_recs} records from {len(results)} papers -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
