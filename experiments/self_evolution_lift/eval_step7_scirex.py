# -*- coding: utf-8 -*-
"""step7 模块2b: SciREX public-dataset n-ary extraction evaluation.

SciREX gold n_ary_relations are {Material, Method, Metric, Task, score} dicts —
exactly our n-ary hypergraph shape (method + object + condition + metric).
Real method (discipline 3): extract_hypergraph on the doc words -> compare the
extracted Method surfaces to gold Method mentions.

metric: Method mention recall (how many gold Method strings appear in the
extracted hypergraph nodes), and n-ary relation overlap (coarse).
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from granular_agent.hypergraph_schema import seed_meta_hypergraph_general
from granular_agent.hypergraph_extractor import extract_hypergraph
from granular_agent.structure_mapper import map_structure

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scirex", "release_data")
if not os.path.isdir(DATA):
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scirex", "release_data")
N = 8  # sample size (extract_hypergraph is slow)


def load_docs():
    return [json.loads(l) for l in open(os.path.join(DATA, "dev.jsonl"), encoding="utf-8")]


def words_to_blocks(doc):
    """SciREX doc words -> block format (one block per sentence)."""
    sents = doc.get("sentences", [])
    words = doc.get("words", [])
    if not sents or not words:
        return []
    blocks = []
    cursor = 0
    # sentences is a list of [start,end] word indices into words
    for i, span in enumerate(sents):
        if isinstance(span, (list, tuple)) and len(span) == 2:
            start, end = span
            text = " ".join(words[start:end]).strip()
        else:
            text = ""
        if len(text) < 10:
            continue
        end_c = cursor + len(text)
        blocks.append({"index": len(blocks), "char_start": cursor, "char_end": end_c, "text": text})
        cursor = end_c + 1
    return blocks


def extract_for_doc(doc):
    blocks = words_to_blocks(doc)
    if len(blocks) < 2:
        return None
    meta = seed_meta_hypergraph_general()
    smap = map_structure(str(doc.get("doc_id")), blocks, llm="deepseek")
    if not smap or not smap.get("dag", {}).get("nodes"):
        return None
    return extract_hypergraph(smap, blocks, meta, llm="deepseek",
                              paper_id=str(doc.get("doc_id")),
                              domain="NLP/ML science", evolve=True)


def gold_methods(doc):
    """All distinct gold Method strings from n_ary_relations."""
    ms = set()
    for rel in doc.get("n_ary_relations", []) or []:
        m = rel.get("Method")
        if m:
            ms.add(m)
    return ms


def extracted_surfaces(res):
    """All node surfaces in the extracted instance hypergraph (lowercased set + raw)."""
    if not res or not res.get("instance"):
        return set()
    inst = res["instance"]
    return {n.surface.lower().strip() for n in inst.nodes.values() if n.surface}


def main():
    docs = load_docs()
    print(f"SciREX dev docs: {len(docs)}", flush=True)
    with_rel = [d for d in docs if d.get("n_ary_relations")]
    print(f"with n_ary_relations: {len(with_rel)}", flush=True)
    sample = with_rel[:N]
    print(f"sample: {N}", flush=True)

    total_gold = 0
    total_hit = 0
    for i, d in enumerate(sample):
        gm = gold_methods(d)
        res = extract_for_doc(d)
        es = extracted_surfaces(res)
        # recall: gold method mentions appearing (case-insensitive substring) in any surface
        hit = 0
        for m in gm:
            ml = m.lower().replace("_", " ")
            if any(ml in s or s in ml for s in es):
                hit += 1
        total_gold += len(gm)
        total_hit += hit
        recall = hit / len(gm) if gm else 0
        n_he = len(res.get("instance", {}).hyperedges) if res and res.get("instance") else 0
        print(f"[{i+1}/{len(sample)}] doc={d.get('doc_id')} gold_methods={len(gm)} "
              f"hit={hit} recall={recall:.2f} extracted_edges={n_he}", flush=True)
        if i == 0 and res and res.get("instance"):
            print("  extracted surfaces (sample):", sorted(list(es))[:8], flush=True)

    recall = total_hit / total_gold if total_gold else 0
    print(f"\n{'='*55}\n=== SciREX Method mention recall (sample N={len(sample)}) ===", flush=True)
    print(f"  gold methods: {total_gold}, hit: {total_hit}, recall: {recall:.3f}", flush=True)
    json.dump({"n": len(sample), "total_gold": total_gold, "total_hit": total_hit, "recall": recall},
              open(os.path.join(os.path.dirname(__file__), "runs", "ARFM2024", "step7_scirex.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
