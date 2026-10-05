# -*- coding: utf-8 -*-
"""Abstract-only coarse records (moved from kb_compiler.records.coarse_extract: same prompt, parsing and record shape).
LLM calls go through compilescholar.llm (local server for the field-state compiler)."""
from __future__ import annotations

from ...llm.client import call_local, call_paratera
from ...llm.jsonparse import parse_json_response


def _route(spec: str):
    if ":" in spec:
        prov, model = spec.split(":", 1)
        return prov, model
    return "paratera", spec


def call_json(prompt: str, model_spec: str, max_tokens: int = 8000, retries: int = 3, salvage: bool = False,
              salvage_key: str = "kind", salvage_wrapper: str = "records"):
    """LLM call -> parsed JSON or None (same retry/salvage behaviour as kb_compiler.records.common.call_json for the
    local and paratera providers). Never raises."""
    from ...llm.jsonparse import salvage_json_records
    prov, model = _route(model_spec)
    fn = call_local if prov == "local" else call_paratera
    for _ in range(retries):
        raw = fn(prompt, model=model, max_tokens=max_tokens, temperature=0.0, enable_thinking=False)
        obj = parse_json_response(raw)
        if obj is not None:
            return obj
        if salvage:
            obj = salvage_json_records(raw, salvage_key, salvage_wrapper)
            if obj:
                return obj
    return None


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
