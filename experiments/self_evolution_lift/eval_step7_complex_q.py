# -*- coding: utf-8 -*-
"""step7 模块1: 综述复杂学科问题评测 (创新评测, schema 路由发力处).

Design: questions that need the higher-order method-relation schema to answer
(limitation-resolution causal chain / direction / method-comparison). A frozen
schema (no lift) cannot answer because it has no METHOD_ nodes / higher_order
patterns; a lifted schema can.

arms:
  F frozen: answer from a schema with NO lift (0 method relations) -> should fail
  B full:   answer from a schema WITH lift (citation+quals) -> should pass

judge: GLM-5 scores whether the answer correctly uses the method-relation.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from granular_agent.hypergraph_schema import seed_meta_hypergraph_general
from granular_agent.hypergraph_lifter import lift_corpus, HIGHER_ORDER_FAMILY
from granular_agent.llm_client import call_paratera, parse_json_response

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "runs"), "ARFM2024", "full")
CIT = os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "runs"), "ARFM2024", "citations.json")
PAPERS = ["Midi_2004","Jop_2006","Pouliquen_1999_Scaling","Bouzid_2013","Bouzid_2015",
          "Kamrin_2012","Kamrin_2015","Jenkins_1983","Jenkins_2010","Gray_2005_A_theory",
          "Gray_2006_Particle-size","Savage_1988_Particle_size","Bazant_2006_The_spot",
          "Tüzün_1979_Experimental","Tripathi_2013_Density","Yoon_2006_The_influence",
          "Sarkar_2008_Experimental","Savage_1998_Analyses","Hill_2014_Segregation"]

# complex questions grounded in gold relations — need the method-relation schema
QUESTIONS = [
    {"q": "I-gradient 模型如何扩展 μ(I) 局部流变? 指出 μ(I) 的局限及 I-gradient 如何解决。",
     "need": "extends: I-gradient extends μ(I) (M17->M1)",
     "src_paper": "Bouzid", "tgt_paper": "Jop"},
    {"q": "非局部颗粒流体性模型(NGF)相对于 μ(I) 本构律的关系是什么? 谁是背景谁继承?",
     "need": "background: μ(I) is background of NGF",
     "src_paper": "Jop", "tgt_paper": "Bouzid"},
    {"q": "Tripathi 密度差驱动分离模型如何改进 Gray 的浅层分离理论?",
     "need": "improves: Tripathi improves Gray (M26->M23)",
     "src_paper": "Tripathi", "tgt_paper": "Gray"},
    {"q": "动理学理论的密堆扩展相对于基础动理学理论是什么关系?",
     "need": "extends: ext-kinetic extends kinetic (M11->M10)",
     "src_paper": "Jenkins_2010", "tgt_paper": "Jenkins_1983"},
]


def find(pre):
    for f in os.listdir(ROOT):
        if f.startswith(pre) and f.endswith(".json"):
            return os.path.join(ROOT, f)
    return None


def build_ebp():
    ebp = {}
    for p in PAPERS:
        fp = find(p)
        if not fp:
            continue
        full = os.path.basename(fp)[:-5]
        d = json.load(open(fp, encoding='utf-8'))
        if d.get('edges'):
            ebp[full] = [{"paper": full, **e} for e in d['edges']]
    return ebp


def schema_relation_summary(meta):
    """Render the higher-order method relations in the schema (empty if frozen)."""
    rels = []
    for pid, pat in meta.patterns.items():
        if pat.family == HIGHER_ORDER_FAMILY:
            rels.append(f"  - {pat.description}")
    return "\n".join(rels) if rels else "  (无方法关系 — schema 未升层)"


def answer(q, schema_ctx, arm_label):
    prompt = f"""你是学科问题解答器。基于下面【schema 方法关系】回答问题。
若 schema 无相关方法关系, 诚实回答"schema 无此信息", 不臆测。
输出严格 JSON (无 markdown 围栏):
{{
  "answer": "回答 (引用 schema 方法关系, 无则写 'schema 无此信息')",
  "used_relation": "用到的关系描述 或 null"
}}

【问题】{q}

【schema 方法关系 ({arm_label})】
{schema_ctx}
"""
    for _ in range(3):
        r = call_paratera(prompt, model="GLM-5-Turbo", max_tokens=300, temperature=0.0)
        obj = parse_json_response(r) if r else None
        if obj:
            return obj
    return {"answer": "(judge failed)", "used_relation": None}


def judge_correctness(q, need, ans):
    """GLM-5 judge: does the answer correctly address the question using a real relation?"""
    prompt = f"""判断【回答】是否正确回答了【问题】, 且用到了【期望关系】。
输出严格 JSON: {{"correct": true|false, "reason": "一句话"}}
【问题】{q}
【期望关系】{need}
【回答】{json.dumps(ans, ensure_ascii=False)}
"""
    for _ in range(3):
        r = call_paratera(prompt, model="GLM-5-Turbo", max_tokens=200, temperature=0.0)
        obj = parse_json_response(r) if r else None
        if obj:
            return obj
    return {"correct": False, "reason": "judge failed"}


def main():
    ebp = build_ebp()
    print(f"corpus: {len(ebp)} papers", flush=True)

    # F frozen: no lift
    metaF = seed_meta_hypergraph_general()
    ctxF = schema_relation_summary(metaF)
    # B full: lift with citation+quals
    pc = {n: list(rec.get("cites_corpus", []))
          for n, rec in json.load(open(CIT, encoding="utf-8")).items()}
    metaB = seed_meta_hypergraph_general()
    lift_corpus(metaB, ebp, paper_citations=pc, use_quals=True)
    ctxB = schema_relation_summary(metaB)

    print(f"\nF frozen schema relations: {ctxF.count(chr(10))}", flush=True)
    print(f"B full schema relations: {ctxB.count(chr(10))}", flush=True)

    f_correct = 0
    b_correct = 0
    print(f"\n{'='*60}", flush=True)
    for i, q in enumerate(QUESTIONS):
        ansF = answer(q["q"], ctxF, "frozen 无升层")
        ansB = answer(q["q"], ctxB, "full 升层")
        jF = judge_correctness(q["q"], q["need"], ansF)
        jB = judge_correctness(q["q"], q["need"], ansB)
        f_correct += int(jF.get("correct"))
        b_correct += int(jB.get("correct"))
        print(f"\n[{i+1}/{len(QUESTIONS)}] Q: {q['q'][:50]}", flush=True)
        print(f"  need: {q['need']}", flush=True)
        print(f"  F frozen: correct={jF.get('correct')} ans={str(ansF.get('answer',''))[:70]!r}", flush=True)
        print(f"  B full:   correct={jB.get('correct')} ans={str(ansB.get('answer',''))[:70]!r}", flush=True)

    n = len(QUESTIONS)
    print(f"\n{'='*60}\n=== 模块1 综述复杂问题准确率 (N={n}) ===", flush=True)
    print(f"  F frozen (无升层): {f_correct}/{n} = {f_correct/n:.3f}", flush=True)
    print(f"  B full (升层):      {b_correct}/{n} = {b_correct/n:.3f}", flush=True)
    json.dump({"n": n, "f_correct": f_correct, "b_correct": b_correct,
               "questions": QUESTIONS},
              open(os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "runs"), "ARFM2024", "step7_complex_q.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
