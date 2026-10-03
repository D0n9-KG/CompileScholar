# -*- coding: utf-8 -*-
"""Field-state compiler: a set of papers -> field-level objects that no single paper states.

叙事 v8 的核心组件。此前的"状态层"（cs2/base_kb_build/build_state_v2 + merge_state_v2）只是把综述作者自己的
taxonomy 抽取结果重新组织——10-03 实测 CS2 状态通道 842 条证据里 841 条与 KB 已有记录逐字相同，消融 Δ≈0；
系统里并没有"从原始论文编译领域状态"的编译器。本模块补上它。

Input: papers [{key, title, year, abstract}]（可选：已抽好的 records，按 key）。
Stages（全部用同一个本地模型；输出的每条领域级事实都带证据 id，指向某篇论文的逐字句子）：
  1. records    : 每篇论文的粗记录（method / finding / limitation，quote=摘要原句）——复用 records.coarse_extract
  2. families   : 方法族归纳——按论文一句话摘要分批提出族（名字 + 定义 + 成员），再跨批合并同义族
  3. family facts: 每个族把成员记录（+ 其他论文对成员的批评/局限）合成为族级
        properties  （做法 / 区别 / 优势 / 适用条件）
        limitations （共同弱点、失效条件、未解决问题）
     每条事实列出支撑的证据 id；n_papers=被引的不同论文数（≥2 即跨论文成立）
  4. area problems: 全部论文的局限/批评陈述跨论文聚合为领域级开放问题（同一问题被几篇论文独立陈述 = n_papers）
Output: {"families": [{id, name, definition, members, properties, limitations}], "problems": [{text, support, n_papers}],
         "records": {key: [...]}, "stats": {...}}
跨族比较关系（relations）暂不编译：领域层检验的金标只有族/性质/局限，没有消费者（按"只保留有消费者的对象"）。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re

from kb_compiler.records.coarse_extract import extract_coarse
from kb_infra.llm import call_local, parse_json_response

MODEL = "Qwen3.8-27B"


def _chat(prompt: str, max_tokens: int = 4000) -> dict | None:
    for t in (0.0, 0.3):
        obj = parse_json_response(call_local(prompt, model=MODEL, max_tokens=max_tokens, temperature=t,
                                              enable_thinking=False) or "")
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------- 1. records
def extract_records(papers: list[dict], workers: int = 8) -> dict[str, list[dict]]:
    def one(p):
        out = extract_coarse(p["title"], p.get("abstract") or "", p.get("year"), "local:" + MODEL)
        return p["key"], out.get("records") or []
    with cf.ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, papers))


def _one_liner(p: dict, recs: list[dict]) -> str:
    m = next((r.get("claim") for r in recs if r.get("kind") == "method" and r.get("claim")), None)
    if not m:
        ab = (p.get("abstract") or "").strip()
        m = re.split(r"(?<=[.!?])\s+", ab)[0] if ab else ""
    return f"{p['title'][:140]} — {m[:220]}"


# ---------------------------------------------------------------- 2. families
PROPOSE = """You are organizing the literature of one research area into method families, the way a survey's taxonomy
groups prior work.

Papers (id: title — what it does):
{lines}

Group these papers into families. A family is a set of papers that share a technical approach, or that address the
same sub-problem in a comparable way. Give each family a short name (as a survey section heading would read) and a
one-sentence definition. Put every paper id in exactly one family. Papers that fit no family with at least one other
paper go into a family named "Other".

Return JSON only: {{"families": [{{"name": "...", "definition": "...", "members": ["p1", "p7"]}}]}}"""

MERGE = """Below are method families proposed separately for different batches of papers from the same research area.
Some families from different batches are the same family under different names. Group the ids of families that should
be merged (same approach / same sub-problem), and give the merged family a name and one-sentence definition.
Every id must appear in exactly one group; a family with no duplicate is a group of one.

Families (id: name — definition [number of papers]):
{lines}

Return JSON only: {{"groups": [{{"ids": ["f1", "f4"], "name": "...", "definition": "..."}}]}}"""


def induce_families(papers: list[dict], records: dict[str, list[dict]], batch: int = 80) -> list[dict]:
    pid = {f"p{i + 1}": p["key"] for i, p in enumerate(papers)}
    lines = [f"p{i + 1}: {_one_liner(p, records.get(p['key']) or [])}" for i, p in enumerate(papers)]
    chunks = [lines[i:i + batch] for i in range(0, len(lines), batch)]

    def propose(chunk):
        obj = _chat(PROPOSE.format(lines="\n".join(chunk)), max_tokens=6000) or {}
        ok = {ln.split(":", 1)[0] for ln in chunk}
        fams = []
        for f in obj.get("families") or []:
            if not isinstance(f, dict) or not f.get("name"):
                continue
            mem = [str(m).strip() for m in f.get("members") or [] if str(m).strip() in ok]
            if mem:
                fams.append({"name": str(f["name"])[:120], "definition": str(f.get("definition") or "")[:300],
                             "members": mem})
        return fams

    with cf.ThreadPoolExecutor(4) as ex:
        proposed = [f for fs in ex.map(propose, chunks) for f in fs]
    other = [f for f in proposed if f["name"].strip().lower() == "other"]
    proposed = [f for f in proposed if f["name"].strip().lower() != "other"]
    if len(chunks) > 1 and proposed:
        fid = {f"f{i + 1}": f for i, f in enumerate(proposed)}
        obj = _chat(MERGE.format(lines="\n".join(f"{k}: {f['name']} — {f['definition']} [{len(f['members'])}]"
                                                 for k, f in fid.items())), max_tokens=6000) or {}
        merged, used = [], set()
        for g in obj.get("groups") or []:
            ids = [i for i in (g.get("ids") or []) if i in fid and i not in used]
            if not ids:
                continue
            used |= set(ids)
            merged.append({"name": str(g.get("name") or fid[ids[0]]["name"])[:120],
                           "definition": str(g.get("definition") or fid[ids[0]]["definition"])[:300],
                           "members": sorted({m for i in ids for m in fid[i]["members"]})})
        merged += [f for k, f in fid.items() if k not in used]  # 合并器漏掉的族原样保留
        proposed = merged
    # 一篇论文只属一个族：重复分配时保留先出现的
    seen, out = set(), []
    for f in proposed:
        mem = [m for m in f["members"] if m not in seen]
        seen |= set(mem)
        if mem:
            out.append({**f, "members": [pid[m] for m in mem]})
    rest = [pid[k] for k in pid if k not in seen]
    if rest or other:
        out.append({"name": "Other", "definition": "papers not grouped with any other paper", "members": rest})
    for i, f in enumerate(out):
        f["id"] = f"F{i + 1}"
    return out


# ---------------------------------------------------------------- 3. family facts
FACTS = """You are writing the family-level facts a survey would state about one method family.

Family: {name} — {definition}

Evidence (verbatim statements from papers; id: [paper title] statement):
{evidence}

Write
- properties: how methods in this family work, what distinguishes them, their strengths, and the conditions under which
  they apply;
- limitations: their weaknesses, failure conditions, unresolved problems and open questions.
Rules: one sentence per fact. Generalize across papers when the evidence supports it, and list every evidence id that
supports the fact. State nothing the evidence does not say. Prefer facts supported by several papers; do not restate a
single paper's contribution as a family property unless it characterizes the family.

Return JSON only: {{"properties": [{{"text": "...", "evidence": ["e3", "e8"]}}], "limitations": [{{"text": "...", "evidence": ["e5"]}}]}}"""


def _txt(r: dict) -> str:
    """证据文本：优先逐字 quote，退回 claim。"""
    return str(r.get("quote") or r.get("claim") or "").strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def family_facts(fam: dict, papers: dict[str, dict], records: dict[str, list[dict]], max_ev: int = 70) -> dict:
    members = set(fam["members"])
    subj = {_norm(r.get("subject")) for k in members for r in records.get(k) or [] if len(_norm(r.get("subject"))) > 2}
    ev = []
    for k in fam["members"]:
        for r in records.get(k) or []:
            if _txt(r):
                ev.append((k, r))
    # 其他论文对本族成员的批评/局限（"mentions" 命中成员方法名）——单篇论文自己不会写自己的这类局限
    for k, recs in records.items():
        if k in members:
            continue
        for r in recs:
            if (r.get("kind") == "limitation" or r.get("claim_type") == "criticism") and _txt(r):
                if any(_norm(m) in subj for m in r.get("mentions") or []):
                    ev.append((k, r))
    ev = ev[:max_ev]
    if not ev:
        return {"properties": [], "limitations": []}
    eid = {f"e{i + 1}": (k, r) for i, (k, r) in enumerate(ev)}
    txt = "\n".join(f"{i}: [{papers[k]['title'][:90]}] {_txt(r)}" for i, (k, r) in eid.items())
    obj = _chat(FACTS.format(name=fam["name"], definition=fam["definition"], evidence=txt), max_tokens=5000) or {}
    out = {}
    for role in ("properties", "limitations"):
        facts = []
        for f in obj.get(role) or []:
            if not isinstance(f, dict) or not str(f.get("text") or "").strip():
                continue
            ids = [str(x).strip("[] ") for x in f.get("evidence") or []]
            sup = [{"paper": eid[i][0], "quote": _txt(eid[i][1])} for i in ids if i in eid]
            if not sup:
                continue  # 没有可解析证据的事实不收（可追溯是硬约束）
            facts.append({"text": str(f["text"]).strip()[:400], "support": sup,
                          "n_papers": len({s["paper"] for s in sup})})
        out[role] = facts
    return out


# ---------------------------------------------------------------- 4. area problems
PROBLEMS = """Below are statements, taken verbatim from paper abstracts in one research area, about limitations, unsolved
problems and shortcomings of existing work.

{evidence}

Group statements that describe the same underlying problem of the area (possibly worded differently, or about different
methods that share the problem). For each group write one sentence stating the problem as a survey's "challenges and
open problems" section would, specific enough to be useful (name the kind of method, data or setting it concerns), and
list every statement id in the group. A statement that shares its problem with no other statement is a group of one.
Every id must appear in exactly one group.

Return JSON only: {{"problems": [{{"text": "...", "evidence": ["s2", "s9"]}}]}}"""


def area_problems(papers: dict[str, dict], records: dict[str, list[dict]], batch: int = 120) -> list[dict]:
    """领域级开放问题：跨全部论文聚合"局限/批评"陈述。
    摘要里的局限句多数是在说"以往工作/本领域做不到 X"（研究动机），不是本文所属方法族自身的局限——
    只挂到族下会把它们归错（10-03 留出综述 dev 抽查：日志解析综述 8 条局限落进跑题的"异常检测"族）。
    这里把它们作为领域对象：同一问题被几篇论文独立陈述（n_papers），即"多篇合在一起才成立"的状态。"""
    items = [(k, r) for k, recs in records.items() for r in recs
             if (r.get("kind") == "limitation" or r.get("claim_type") == "criticism") and _txt(r)]
    out = []
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        sid = {f"s{i + 1}": x for i, x in enumerate(chunk)}
        txt = "\n".join(f"{i}: {_txt(r)}" for i, (k, r) in sid.items())
        obj = _chat(PROBLEMS.format(evidence=txt), max_tokens=6000) or {}
        used = set()
        for p in obj.get("problems") or []:
            if not isinstance(p, dict) or not str(p.get("text") or "").strip():
                continue
            ids = [str(x).strip("[] ") for x in p.get("evidence") or [] if str(x).strip("[] ") in sid]
            ids = [i for i in ids if i not in used]
            if not ids:
                continue
            used |= set(ids)
            sup = [{"paper": sid[i][0], "quote": _txt(sid[i][1])} for i in ids]
            out.append({"text": str(p["text"]).strip()[:400], "support": sup,
                        "n_papers": len({s["paper"] for s in sup})})
        for i, (k, r) in sid.items():  # 分组器漏掉的陈述原样保留为单条问题（不丢证据）
            if i not in used:
                out.append({"text": _txt(r)[:400], "support": [{"paper": k, "quote": _txt(r)}], "n_papers": 1})
    out.sort(key=lambda p: -p["n_papers"])
    return out


# ---------------------------------------------------------------- entry
def compile_field_state(papers: list[dict], records: dict[str, list[dict]] | None = None, workers: int = 8) -> dict:
    papers = [p for p in papers if p.get("title")]
    by_key = {p["key"]: p for p in papers}
    if records is None:
        records = extract_records([p for p in papers if (p.get("abstract") or "").strip()], workers=workers)
    fams = induce_families(papers, records)
    with cf.ThreadPoolExecutor(4) as ex:
        facts = list(ex.map(lambda f: family_facts(f, by_key, records) if f["name"] != "Other"
                            else {"properties": [], "limitations": []}, fams))
    for f, x in zip(fams, facts):
        f.update(x)
    problems = area_problems(by_key, records)
    n_lim = [l for f in fams for l in f["limitations"]]
    stats = {"papers": len(papers), "with_records": sum(1 for v in records.values() if v),
             "records": sum(len(v) for v in records.values()), "families": sum(1 for f in fams if f["name"] != "Other"),
             "other_members": sum(len(f["members"]) for f in fams if f["name"] == "Other"),
             "properties": sum(len(f["properties"]) for f in fams), "limitations": len(n_lim),
             "cross_paper_limitations": sum(1 for l in n_lim if l["n_papers"] >= 2),
             "area_problems": len(problems), "cross_paper_problems": sum(1 for p in problems if p["n_papers"] >= 2)}
    return {"families": fams, "problems": problems, "records": records, "stats": stats}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ps = json.load(open(sys.argv[1], encoding="utf-8"))
    st = compile_field_state(ps)
    json.dump(st, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(st["stats"])
