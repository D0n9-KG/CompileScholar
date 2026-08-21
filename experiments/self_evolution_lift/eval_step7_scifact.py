# -*- coding: utf-8 -*-
"""step7 模块2: SciFact public-dataset fact-checking via our hypergraph extraction.

Real method (discipline 3): for each claim with gold SUPPORT evidence, take its
cited doc, run extract_hypergraph on the abstract (blocks from sentences), then
ask a GLM-5 judge whether the extracted hypergraph evidence supports the claim.

Compares:
  - hypergraph-arm: judge sees the extracted n-ary relations + their evidence spans
  - baseline (naive-RAG): judge sees the raw abstract sentences only

Metrics: SUPPORT accuracy (does the arm correctly identify SUPPORT claims?).
This is the external public-dataset result (module 2). Small sample (N=12) for
feasibility; methodology is sound and reproducible to full 188.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from granular_agent.hypergraph_schema import seed_meta_hypergraph_general
from granular_agent.hypergraph_extractor import extract_hypergraph
from granular_agent.structure_mapper import map_structure
from granular_agent.llm_client import call_paratera, parse_json_response

DATA = os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "data")
N = 12  # sample size


def load_corpus():
    return {json.loads(l)["doc_id"]: json.loads(l)
            for l in open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8")}


def load_claims():
    return [json.loads(l) for l in open(os.path.join(DATA, "claims_dev.jsonl"), encoding="utf-8")]


def abstract_to_blocks(doc):
    """SciFact abstract (sentence list) -> block format {index,char_start,char_end,text}."""
    blocks = []
    cursor = 0
    for i, sent in enumerate(doc.get("abstract", [])):
        t = sent.strip()
        if not t:
            continue
        end = cursor + len(t)
        blocks.append({"index": len(blocks), "char_start": cursor, "char_end": end, "text": t})
        cursor = end + 1
    return blocks


def extract_for_doc(doc_id, doc):
    """Run map_structure + extract_hypergraph on one SciFact doc abstract."""
    blocks = abstract_to_blocks(doc)
    if len(blocks) < 2:
        return None
    meta = seed_meta_hypergraph_general()
    smap = map_structure(str(doc_id), blocks, llm="deepseek")
    if not smap or not smap.get("dag", {}).get("nodes"):
        return None
    res = extract_hypergraph(smap, blocks, meta, llm="deepseek", paper_id=str(doc_id),
                             domain="biomedical science", evolve=True,
                             propagate_intra_dag=True)
    return res


def hypergraph_evidence_summary(res):
    """Render the extracted hypergraph edges + evidence for the judge."""
    if not res:
        return "(extraction failed)"
    inst = res.get("instance")
    if not inst:
        return "(no instance)"
    lines = []
    for eid, he in list(inst.hyperedges.items())[:12]:
        nodes = ", ".join(
            (inst.nodes[nid].surface if nid in inst.nodes else nid)
            for nid in he.node_ids)
        ev = (he.evidence_span or "")[:150]
        lines.append(f"  - rel={he.pattern_type} nodes=[{nodes}] ev=\"{ev}\"")
    return "\n".join(lines) if lines else "(no relations extracted)"


def judge_support(claim, context, arm_label):
    """GLM-5 judge: does context support the claim? -> {label, rationale}."""
    prompt = f"""你是科学事实核查判断器。判断下面【上下文】是否支持【claim】。
只基于上下文, 不凭外部知识。输出严格 JSON (无 markdown 围栏):
{{
  "label": "SUPPORT" | "REFUTE" | "NOINFO",
  "rationale": "一句话依据"
}}

【claim】{claim}

【上下文 ({arm_label})】
{context}
"""
    for _ in range(3):
        r = call_paratera(prompt, model="GLM-5-Turbo", max_tokens=200, temperature=0.0)
        obj = parse_json_response(r) if r else None
        if obj:
            return obj
    return {"label": "NOINFO", "rationale": "judge failed"}


def main():
    corpus = load_corpus()
    claims = load_claims()
    # sample N claims with SUPPORT gold (cited doc in corpus)
    sup = [c for c in claims if c.get("evidence") and c.get("cited_doc_ids")
           and c["cited_doc_ids"][0] in corpus]
    sample = sup[:N]
    print(f"sample: {len(sample)} SUPPORT claims (of {len(sup)} available)", flush=True)

    hg_correct = 0   # hypergraph arm correctly says SUPPORT
    rag_correct = 0  # naive-RAG arm correctly says SUPPORT
    cached = {}
    for i, c in enumerate(sample):
        claim = c["claim"]
        doc_id = c["cited_doc_ids"][0]
        doc = corpus[doc_id]
        # hypergraph arm
        res = cached.get(doc_id)
        if res is None:
            res = extract_for_doc(doc_id, doc)
            cached[doc_id] = res
        hg_ctx = hypergraph_evidence_summary(res)
        hg = judge_support(claim, hg_ctx, "extracted hypergraph relations + evidence")
        # naive-RAG arm: raw abstract sentences
        rag_ctx = "\n".join(f"  {s}" for s in doc.get("abstract", [])[:12])
        rag = judge_support(claim, rag_ctx, "raw abstract sentences")
        hg_ok = hg.get("label") == "SUPPORT"
        rag_ok = rag.get("label") == "SUPPORT"
        hg_correct += int(hg_ok)
        rag_correct += int(rag_ok)
        print(f"[{i+1}/{len(sample)}] doc={doc_id} claim={claim[:55]!r}", flush=True)
        print(f"   hypergraph: {hg.get('label')} ({'OK' if hg_ok else 'miss'}) "
              f"rag: {rag.get('label')} ({'OK' if rag_ok else 'miss'})", flush=True)

    print(f"\n{'='*55}\n=== SciFact SUPPORT accuracy (sample N={len(sample)}) ===", flush=True)
    print(f"  hypergraph arm: {hg_correct}/{len(sample)} = {hg_correct/len(sample):.3f}", flush=True)
    print(f"  naive-RAG arm:  {rag_correct}/{len(sample)} = {rag_correct/len(sample):.3f}", flush=True)
    json.dump({"n": len(sample), "hg_correct": hg_correct, "rag_correct": rag_correct},
              open(os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "runs"), "ARFM2024", "step7_scifact.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
