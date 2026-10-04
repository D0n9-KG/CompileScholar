# -*- coding: utf-8 -*-
"""领域层直接检验（叙事 v8 §3.3）：只给系统一篇留出综述所引用的论文（标题+摘要），编译领域状态，
与综述作者自己写下的判断对照。

金标：base_kb_v2/survey_gold.json（综述原文抽取：taxonomy 族名 / 族级性质 / challenges+缺口）。综述本身不在输入里。
三个系统，输入完全相同（引用论文的标题+摘要），同一 27B：
  state  —— src/kb_compiler/views/field_state.compile_field_state（逐篇记录 → 族归纳 → 族级事实，每条事实带逐字证据）
  direct —— 一次调用：把全部论文给 27B，直接写出方法族 + 各族性质/局限 + 领域开放问题（"让 LLM 直接写综述骨架"）
  flat   —— 只做逐篇抽取，不做跨论文组织：全部 limitation 记录原样作为"局限"，method 记录主语作为"族"
判分：DeepSeek-V4.1-Flash（与生成模型不同）逐条判"金标陈述是否被某条系统陈述表达"——
  族召回、性质召回、局限召回；以及系统条目里命中任一金标的比例（"命中率"，精度下界：综述没写的不一定错）。
条目数差异很大（flat 往往条目最多），同时报条目数；另报 recall@K（只取系统前 K 条，K=金标条数）控制数量。
用法：python heldout_field_eval.py [--surveys N] [--systems state,direct,flat]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "base_kb_v2")
REFS = os.path.join(V2, "heldout_refs")
OUT = os.path.join(V2, "heldout_eval")
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "src")))
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",192.168.199.73,localhost,127.0.0.1"

from kb_compiler.views import field_state as FS  # noqa: E402
from kb_infra.llm import call_paratera, parse_json_response  # noqa: E402

JUDGE = os.environ.get("FIELD_JUDGE", "DeepSeek-V4.1-Flash")
MIN_PAPERS = 15


# ---------------------------------------------------------------- systems
DIRECT = """You are given the papers cited by a survey of one research area (title: abstract excerpt).

{lines}

Write the field state of this area as a survey's taxonomy would: the method families (each with a short name and a
one-sentence definition), and for each family its properties (how the methods work, what distinguishes them, strengths,
conditions where they apply) and its limitations (weaknesses, failure conditions, open problems). Also list open
problems of the area as a whole. One sentence per fact. Use only what the papers say.

Return JSON only: {{"families": [{{"name": "...", "definition": "...", "properties": ["..."], "limitations": ["..."]}}],
"open_problems": ["..."]}}"""


def run_direct(papers):
    lines = "\n".join(f"- {p['title'][:150]}: {(p.get('abstract') or '')[:600]}" for p in papers)
    obj = FS._chat(DIRECT.format(lines=lines), max_tokens=8000) or {}
    fams = []
    for f in obj.get("families") or []:
        if isinstance(f, dict) and f.get("name"):
            fams.append({"name": str(f["name"]), "definition": str(f.get("definition") or ""),
                         "properties": [{"text": str(x)} for x in f.get("properties") or [] if str(x).strip()],
                         "limitations": [{"text": str(x)} for x in f.get("limitations") or [] if str(x).strip()]})
    op = [{"text": str(x)} for x in obj.get("open_problems") or [] if str(x).strip()]
    if op:
        fams.append({"name": "Open problems of the area", "definition": "", "properties": [], "limitations": op})
    return {"families": fams}


MEMORY = """Write the field state of the research area covered by a survey titled "{title}" ({year}), as that survey's
taxonomy would: the method families (each with a short name and a one-sentence definition), and for each family its
properties (how the methods work, what distinguishes them, strengths, conditions where they apply) and its limitations
(weaknesses, failure conditions, open problems). Also list open problems of the area as a whole. One sentence per fact.

Return JSON only: {{"families": [{{"name": "...", "definition": "...", "properties": ["..."], "limitations": ["..."]}}],
"open_problems": ["..."]}}"""


def run_memory(g):
    """记忆对照：不给任何论文，只给综述标题——测 27B 参数记忆能恢复多少（留出综述 2020-2024，模型多半见过）。
    direct 与 memory 接近 = 该检验在这批综述上主要测记忆，而非从论文编译的能力。"""
    obj = FS._chat(MEMORY.format(title=g["title"], year=g.get("year")), max_tokens=8000) or {}
    fams = []
    for f in obj.get("families") or []:
        if isinstance(f, dict) and f.get("name"):
            fams.append({"name": str(f["name"]), "definition": str(f.get("definition") or ""),
                         "properties": [{"text": str(x)} for x in f.get("properties") or [] if str(x).strip()],
                         "limitations": [{"text": str(x)} for x in f.get("limitations") or [] if str(x).strip()]})
    op = [{"text": str(x)} for x in obj.get("open_problems") or [] if str(x).strip()]
    if op:
        fams.append({"name": "Open problems of the area", "definition": "", "properties": [], "limitations": op})
    return {"families": fams}


def run_flat(records):
    fams = {}
    for k, recs in records.items():
        for r in recs:
            if r["kind"] == "method" and r.get("subject"):
                fams.setdefault(r["subject"].strip().lower(), {"name": r["subject"].strip(), "definition": r["claim"],
                                                               "properties": [], "limitations": []})
    lims = [{"text": r["claim"]} for recs in records.values() for r in recs if r["kind"] == "limitation"]
    props = [{"text": r["claim"]} for recs in records.values() for r in recs if r["kind"] in ("method", "finding")]
    out = list(fams.values())
    out.append({"name": "(all records)", "definition": "", "properties": props, "limitations": lims})
    return {"families": out}


# ---------------------------------------------------------------- judge
MATCH = """Gold statements were written by the expert authors of a survey. Candidate statements were produced by a system
that read only the papers the survey cites. For each gold statement, list the ids of candidates that express the same
point — the same {what}.

A candidate matches only if a reader of the candidate alone would learn the gold point: it must be about the same
method or object AND state the same property, weakness or problem. Examples of NON-matches:
- gold "k-means is sensitive to initialization" vs candidate "k-means is slow on large datasets" (same method,
  different weakness);
- gold "beam search produces repetitive outputs" vs candidate "random forests overfit noisy labels" (both are
  limitations, different objects);
- any generic statement ("robustness remains an open challenge") for a specific gold point.
Most gold statements will have no matching candidate; return an empty list for them.

Gold:
{gold}

Candidates:
{cand}

Return JSON only: {{"matches": {{"G1": ["C4"], "G2": []}}}}"""


def judge_match(gold: list[str], cand: list[str], what: str) -> dict[str, list[str]]:
    if not gold or not cand:
        return {f"G{i + 1}": [] for i in range(len(gold))}
    res = {}
    for gs in range(0, len(gold), 25):
        g = gold[gs:gs + 25]
        gtxt = "\n".join(f"G{gs + i + 1}: {t[:300]}" for i, t in enumerate(g))
        ctxt = "\n".join(f"C{i + 1}: {t[:300]}" for i, t in enumerate(cand[:250]))
        obj = None
        for _ in range(3):
            obj = parse_json_response(call_paratera(MATCH.format(what=what, gold=gtxt, cand=ctxt), model=JUDGE,
                                                    max_tokens=4000, enable_thinking=False) or "")
            if isinstance(obj, dict) and isinstance(obj.get("matches"), dict):
                break
        m = (obj or {}).get("matches") or {}
        for i in range(len(g)):
            k = f"G{gs + i + 1}"
            res[k] = [c for c in (m.get(k) or []) if isinstance(c, str) and re.fullmatch(r"C\d+", c)]
    return res


def score(gold: dict, sys_state: dict) -> dict:
    fams = [f for f in sys_state["families"] if f["name"] not in ("Other", "(all records)")]
    cand_f = [f"{f['name']} — {f.get('definition', '')}" for f in fams]
    cand_p = [p["text"] for f in sys_state["families"] for p in f.get("properties") or []]
    cand_l = [l["text"] for f in sys_state["families"] for l in f.get("limitations") or []]
    # 领域级开放问题（field_state v2）：按跨论文支持数排序在前，与族级局限一起作为"局限"候选
    cand_l = [p["text"] for p in sys_state.get("problems") or []] + cand_l
    g_f = gold["families"]
    g_p = [p["text"] for p in gold["properties"]]
    g_l = [l["text"] for l in gold["limitations"]]
    out = {}
    for name, g, c, what in (("families", g_f, cand_f, "method family / sub-topic of the area"),
                             ("properties", g_p, cand_p, "property of the same kind of method"),
                             ("limitations", g_l, cand_l, "limitation, challenge or open problem")):
        m = judge_match(g, c, what)
        hit_c = {x for v in m.values() for x in v}
        k = len(g)
        rec_at_k = (sum(1 for v in m.values() if any(int(x[1:]) <= k for x in v)) / k) if k else None
        out[name] = {"n_gold": len(g), "n_cand": len(c),
                     "recall": (sum(1 for v in m.values() if v) / len(g)) if g else None,
                     "recall_at_k": rec_at_k,
                     "cand_hit_rate": (len(hit_c) / len(c)) if c else None, "matches": m}
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surveys", type=int, default=0)
    ap.add_argument("--systems", default="state,state_v2,direct,flat,memory")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gold = json.load(open(os.path.join(V2, "survey_gold.json"), encoding="utf-8"))
    todo = []
    for pid, g in gold.items():
        p = os.path.join(REFS, f"{pid}.json")
        if not os.path.exists(p):
            continue
        papers = [r for r in json.load(open(p, encoding="utf-8")) if (r.get("abstract") or "").strip()]
        if len(papers) >= MIN_PAPERS and (g["limitations"] or g["properties"]):
            todo.append((pid, g, papers))
    if a.surveys:
        todo = todo[:a.surveys]
    print(f"[field-eval] surveys {len(todo)} | systems {a.systems}", flush=True)
    systems = a.systems.split(",")
    rows = {}
    for pid, g, papers in todo:
        sp = os.path.join(OUT, f"{pid}.state.json")
        if os.path.exists(sp):
            st = json.load(open(sp, encoding="utf-8"))
        else:
            st = FS.compile_field_state(papers)
            json.dump(st, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        outs = {"state": st}
        if "state_v2" in systems:
            # 同一批逐篇记录（与 state/flat 完全相同的抽取结果），只换组织层——隔离"组织"本身的效果
            p2 = os.path.join(OUT, f"{pid}.state_v2.json")
            if not os.path.exists(p2):
                json.dump(FS.compile_field_state(papers, records=st["records"]), open(p2, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            outs["state_v2"] = json.load(open(p2, encoding="utf-8"))
        if "direct" in systems:
            dp = os.path.join(OUT, f"{pid}.direct.json")
            if not os.path.exists(dp):
                json.dump(run_direct(papers), open(dp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            outs["direct"] = json.load(open(dp, encoding="utf-8"))
        if "flat" in systems:
            outs["flat"] = run_flat(st["records"])
        if "memory" in systems:
            mp = os.path.join(OUT, f"{pid}.memory.json")
            if not os.path.exists(mp):
                json.dump(run_memory(g), open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            outs["memory"] = json.load(open(mp, encoding="utf-8"))
        scp = os.path.join(OUT, f"{pid}.scores.json")
        prev = json.load(open(scp, encoding="utf-8")) if os.path.exists(scp) else {}
        with cf.ThreadPoolExecutor(3) as ex:
            fut = {s: ex.submit(score, g, outs[s]) for s in systems if s not in prev}
            for s, f in fut.items():
                prev[s] = f.result()
        json.dump(prev, open(scp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        rows[pid] = prev
        line = " | ".join(f"{s}: F {prev[s]['families']['recall'] or 0:.2f} P {prev[s]['properties']['recall'] or 0:.2f} "
                          f"L {prev[s]['limitations']['recall'] or 0:.2f}@{prev[s]['limitations']['n_cand']}"
                          for s in systems)
        print(f"{pid} papers={len(papers)} state={st['stats']} || {line}", flush=True)
    summarize(rows, systems, gold)


def summarize(rows, systems, gold):
    """宏平均。另报 clean 金标口径（clean_survey_gold.py 在看结果前标注；复用已有匹配，不重判）。"""
    for s in systems:
        agg = {}
        for part in ("properties", "limitations"):
            v = []
            for pid, r in rows.items():
                if not r.get(s):
                    continue
                items = gold[pid][part]
                idx = [i for i, x in enumerate(items) if x.get("clean")]
                if not idx:
                    continue
                m = r[s][part]["matches"]
                v.append(sum(1 for i in idx if m.get(f"G{i + 1}")) / len(idx))
            agg[f"{part}.recall_clean"] = round(sum(v) / len(v), 3) if v else None
        print(s, "clean", json.dumps(agg))
    # 汇总（宏平均，跳过金标为空的项）
    for s in systems:
        agg = {}
        for part in ("families", "properties", "limitations"):
            for m in ("recall", "recall_at_k", "cand_hit_rate", "n_cand"):
                v = [r[s][part][m] for r in rows.values() if r.get(s) and r[s][part][m] is not None
                     and r[s][part]["n_gold"]]
                agg[f"{part}.{m}"] = round(sum(v) / len(v), 3) if v else None
        print(s, json.dumps(agg))


if __name__ == "__main__":
    main()
