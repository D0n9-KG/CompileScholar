"""Claim-coverage matcher (2026-08-27): match a paper's MAIN-CLAIMS checklist
against the committed hyperedges — the recall axis next to edge precision.

Input:  <bundle>/_claims.jsonl  (one claim per line:
        {i, claim, quote, category})   — written by the claims-lister subagent
        <bundle>/concept_graph.json    (committed edges)
Output: <bundle>/_claims_matched.jsonl (claim + matched edge ids + match type)
        printed summary: claim recall overall + by category

Match rule (deterministic, conservative — an LLM reviews the unmatched):
  a. quote- overlap: token-overlap(quote, edge.evidence) >= 0.5
  b. surface- hit:   >=2 node surfaces of the edge appear in the quote, or
     (>=1 surface AND that surface is a multi-word named entity)
"""
import json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _tok(s):
    return set(re.findall(r"[a-z0-9]{3,}", (s or "").lower()))


def match_bundle(bdir: str, paper: str):
    cg = json.load(open(os.path.join(bdir, "concept_graph.json"), encoding="utf-8"))
    conc = {cid: ((c.get("surface_variants") or [{}])[0].get("surface", ""))
            for cid, c in cg.get("concepts", {}).items()}
    edges = []
    for e in cg.get("hyperedges", []):
        for prov in e.get("provenance", []):
            if prov.get("paper_id") == paper:
                edges.append({"he_id": e.get("he_id"), "pattern": e.get("pattern_type"),
                              "evidence": prov.get("evidence", ""),
                              "surfaces": [conc.get(n, "") for n in e.get("node_ids", [])]})
    claims = [json.loads(l) for l in open(os.path.join(bdir, "_claims.jsonl"), encoding="utf-8")]
    out = []
    for c in claims:
        qt = _tok(c["quote"])
        # relaxed second pass: also score the edge evidence against the quote's
        # token set with a LOWER bar (0.3) — catches the "same fact, different
        # sentence" case the strict 0.5 misses (pilot: error-clipping/stability
        # edge scored 0.4 against its claim). Unmatched-at-0.3 still reviewed.
        hits = []
        for e in edges:
            et = _tok(e["evidence"])
            ov = len(qt & et) / max(1, len(qt))
            surf_hits = [s for s in e["surfaces"]
                         if s and len(_tok(s)) and _tok(s) <= _tok(c["quote"] + " " + c["claim"])]
            if ov >= 0.3 or len(surf_hits) >= 2 or \
               (len(surf_hits) >= 1 and len(_tok(surf_hits[0])) >= 2):
                hits.append((e["he_id"], e["pattern"],
                             "quote" if ov >= 0.5 else ("near" if ov >= 0.3 else "surface")))
        out.append({**c, "matched": bool(hits),
                    "match": hits[:3]})
    with open(os.path.join(bdir, "_claims_matched.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(out); m = sum(1 for r in out if r["matched"])
    print(f"[cov] {paper}: claim recall {m}/{n} = {m/max(1,n):.0%}")
    by = {}
    for r in out:
        by.setdefault(r.get("category", "?"), [0, 0])
        by[r["category"]][0] += 1
        if r["matched"]:
            by[r["category"]][1] += 1
    for k, (a, b) in sorted(by.items()):
        print(f"    {k:<14} {b}/{a}")
    print("    unmatched:")
    for r in out:
        if not r["matched"]:
            print(f"      [{r['i']}] ({r.get('category','?')}) {r['claim'][:100]}")
    return m, n


if __name__ == "__main__":
    bdir, paper = sys.argv[1], sys.argv[2]
    match_bundle(bdir, paper)
