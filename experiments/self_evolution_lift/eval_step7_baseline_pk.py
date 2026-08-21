# -*- coding: utf-8 -*-
"""step7 模块3: SOTA baseline fair PK — IncSchema-style direct-probe vs our lift.

IncSchema (raspberryice/inc-schema) induces schema by DIRECTLY probing the LLM
for high-order relations ("list extensions/improvements of method X") — it does
NOT lift from low-order edges. Its prompts.py uses expansion prompts like
"What happens after X", "list consequences of X".

We adapt IncSchema's direct-probe style to OUR task (method extends/improves/
compares) and run it with OUR deepseek LLM on the SAME ARFM2024 corpus
(discipline 6: same LLM, same data, controlled). This is the fair PK the goal
requires — not the original IncSchema (which is OpenAI-bound + event-schema).

arms:
  Inc-direct: deepseek directly probes method relations from paper abstracts
              (IncSchema-style: no low-order extraction, no lift)
  ours-lift:  lift_corpus (cluster -> induce -> judge with structural signals)

metric: gold type coverage + relation count (same as ablation).
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from collections import Counter
from granular_agent.hypergraph_schema import seed_meta_hypergraph_general
from granular_agent.hypergraph_lifter import lift_corpus, METHOD_NS, HIGHER_ORDER_FAMILY
from granular_agent.llm_client import call_llm, parse_json_response

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "ARFM2024", "full")
CIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "ARFM2024", "citations.json")
PAPERS = ["Midi_2004","Jop_2006","Pouliquen_1999_Scaling","Bouzid_2013","Bouzid_2015",
          "Kamrin_2012","Kamrin_2015","Jenkins_1983","Jenkins_2010","Gray_2005_A_theory",
          "Gray_2006_Particle-size","Savage_1988_Particle_size","Bazant_2006_The_spot",
          "Tüzün_1979_Experimental","Tripathi_2013_Density","Yoon_2006_The_influence",
          "Sarkar_2008_Experimental","Savage_1998_Analyses","Hill_2014_Segregation"]
GOLD_TYPES = ["extends","improves","compares"]

# IncSchema-style direct-probe prompt: ask the LLM directly for method relations,
# no low-order extraction, no lift. Adapted from IncSchema expansion prompts.
INC_PROBE_PROMPT = """你是科学方法演化关系抽取器。下面是 {npaper} 篇颗粒流论文的标题和摘要。
直接从论文内容判断方法间关系（extends/improves/compares/background），不要经过低阶抽取。

论文:
{papers}

直接列出你判断到的方法间关系（基于论文内容, 不臆测）。输出严格 JSON (无 markdown 围栏):
{{
  "method_relations": [
    {{"src": "方法A", "relation": "extends|improves|compares|background", "tgt": "方法B", "rationale": "依据"}}
  ]
}}
"""


def find(pre):
    for f in os.listdir(ROOT):
        if f.startswith(pre) and f.endswith(".json"):
            return os.path.join(ROOT, f)
    return None


def load_paper_abstracts():
    """Get title + a short text per paper (from the lift-edge evidence, as abstract proxy)."""
    papers = []
    for p in PAPERS:
        fp = find(p)
        if not fp:
            continue
        full = os.path.basename(fp)[:-5]
        d = json.load(open(fp, encoding='utf-8'))
        # use the shortest 6 evidence spans as the paper's textual content proxy
        evs = sorted([e.get('ev', '') for e in d.get('edges', []) if e.get('ev')],
                     key=len)[:6]
        text = " | ".join(evs)[:500]
        papers.append((full, text))
    return papers


def run_inc_direct(papers):
    """IncSchema-style: one LLM call directly probing method relations."""
    body = "\n".join(f"[{i+1}] {name}: {text}" for i, (name, text) in enumerate(papers))
    prompt = INC_PROBE_PROMPT.format(npaper=len(papers), papers=body)
    obj = None
    for _ in range(3):
        r = call_llm(prompt, model="deepseek-chat", max_tokens=4000, temperature=0.0)
        obj = parse_json_response(r) if r else None
        if obj:
            break
    rels = (obj or {}).get("method_relations", []) if isinstance(obj, dict) else []
    by_type = Counter(r.get("relation") for r in rels if r.get("relation") and r.get("relation") != "null")
    types_present = set(by_type.keys())
    gold_hit = sum(1 for g in GOLD_TYPES if g in types_present)
    return {"relations": len(rels), "types": dict(by_type), "gold_hit": gold_hit,
            "method_relations": rels}


def run_ours_lift(ebp, pc):
    meta = seed_meta_hypergraph_general()
    out = lift_corpus(meta, ebp, paper_citations=pc, use_quals=True)
    wr = out['written']['relation_patterns']
    by_type = Counter(r['relation'] for r in wr)
    types_present = set(by_type.keys())
    gold_hit = sum(1 for g in GOLD_TYPES if g in types_present)
    return {"relations": len(wr), "types": dict(by_type), "gold_hit": gold_hit,
            "families": len(out['clusters'])}


def main():
    papers = load_paper_abstracts()
    print(f"corpus: {len(papers)} papers (abstract proxies)", flush=True)
    ebp = {}
    for p in PAPERS:
        fp = find(p)
        if not fp:
            continue
        full = os.path.basename(fp)[:-5]
        d = json.load(open(fp, encoding='utf-8'))
        if d.get('edges'):
            ebp[full] = [{"paper": full, **e} for e in d['edges']]
    pc = {n: list(rec.get("cites_corpus", []))
          for n, rec in json.load(open(CIT, encoding="utf-8")).items()}

    print("\n=== arm IncSchema-direct (deepseek direct probe, no lift) ===", flush=True)
    inc = run_inc_direct(papers)
    print(f"  relations={inc['relations']} types={inc['types']} gold={inc['gold_hit']}/3", flush=True)
    for r in inc['method_relations'][:8]:
        print(f"    [{r.get('relation')}] {str(r.get('src',''))[:25]} -> {str(r.get('tgt',''))[:25]}", flush=True)

    print("\n=== arm ours-lift (cluster+induce+judge, citation+quals) ===", flush=True)
    ours = run_ours_lift(ebp, pc)
    print(f"  families={ours.get('families')} relations={ours['relations']} "
          f"types={ours['types']} gold={ours['gold_hit']}/3", flush=True)

    print(f"\n{'='*60}\n=== 模块3 BASELINE PK (same deepseek LLM, same ARFM2024) ===", flush=True)
    print(f"  IncSchema-style direct-probe: relations={inc['relations']} gold={inc['gold_hit']}/3 types={inc['types']}", flush=True)
    print(f"  ours-lift (low-order lift):   relations={ours['relations']} gold={ours['gold_hit']}/3 types={ours['types']}", flush=True)
    json.dump({"inc_direct": inc, "ours_lift": ours},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "ARFM2024", "step7_baseline_pk.json"), "w"),
              ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
