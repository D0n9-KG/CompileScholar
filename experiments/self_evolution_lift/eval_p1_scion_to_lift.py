# -*- coding: utf-8 -*-
"""把 SCION graph_edges 转成 eval_err_edges 期望的 lift 格式 (methods+relations).
SCION 边: node_1.name/node_2.name/relationship(描述句) → type 用关键词映射+LLM兜底.
产出: scion_lift.json (methods[].name + relations[].{src,tgt,relation})
"""
import os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from granular_agent.llm_client import call_paratera, parse_json_response

TYPE_KEYWORDS = {
    "extends": [r"\bextend", r"generaliz", r"built on", r"based on", r"推广", r"扩展"],
    "improves": [r"\bimprov", r"better", r"more accurate", r"resolv", r"overcome", r"改进"],
    "compares": [r"\bcompar", r"contrast", r"differ", r"对比"],
    "replaces": [r"\breplac", r"supersede", r"instead of", r"取代"],
    "adapts": [r"\badapt", r"apply.{0,15}to", r"适用"],
    "background": [r"background", r"motivat", r"inspire", r"背景", r"启发"],
}


def map_type(relationship):
    """关键词映射 SCION relationship 描述→type."""
    r = relationship.lower()
    for typ, pats in TYPE_KEYWORDS.items():
        if any(re.search(p, r) for p in pats):
            return typ
    return "background"  # 默认


def llm_map_type(edges):
    """LLM 批量映射 SCION relationship→type (兜底关键词)."""
    if not edges:
        return {}
    items = [{"i": i, "r": e["relationship"][:200]} for i, e in enumerate(edges)]
    prompt = """把下面每条"关系描述句"归类到方法演化关系类型. 只基于描述句语义判:
- extends: A在B基础上推广/扩展
- improves: A改进B精度/适用范围, 解决B局限
- compares: A与B比较(并列/不同方法)
- replaces: A取代B
- adapts: A把B适配到新场景
- background: A以B为背景/动机

描述句:
""" + "\n".join(f"{x['i']}: {x['r']}" for x in items) + """

输出JSON: {"mapping": [{"i": 0, "type": "extends"}, ...]}
"""
    resp = call_paratera(prompt, model="GLM-5-Turbo", max_tokens=2000, temperature=0.0)
    o = (parse_json_response(resp) or {}).get("mapping", [])
    return {x["i"]: x["type"] for x in o if "type" in x}


def main():
    edges = json.load(open(os.path.join(os.path.dirname(__file__), "paper_scion/data/output/graph_edges_en.json"), encoding='utf-8'))
    print(f"SCION edges: {len(edges)}")

    # LLM 映射 type
    print("LLM mapping type...")
    llm_map = llm_map_type(edges)
    print(f"LLM mapped: {len(llm_map)}/{len(edges)}")

    methods = []
    seen = set()
    relations = []
    for i, e in enumerate(edges):
        s = e["node_1"]["name"].strip()
        t = e["node_2"]["name"].strip()
        if not s or not t or s == t:
            continue
        # 去重方法
        for nm in [s, t]:
            k = nm.lower()
            if k not in seen:
                seen.add(k)
                methods.append({"name": nm})
        typ = llm_map.get(i) or map_type(e["relationship"])
        relations.append({"src": s, "tgt": t, "relation": typ, "rationale": e["relationship"][:150]})

    print(f"methods: {len(methods)}, relations: {len(relations)}")
    from collections import Counter
    print("type分布:", dict(Counter(r["relation"] for r in relations)))

    out = {"methods": methods, "relations": relations, "n_papers": 19}
    outp = os.path.join(os.path.dirname(__file__), "runs/ARFM2024/arms_err/scion_baseline.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
