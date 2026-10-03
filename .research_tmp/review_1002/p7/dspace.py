"""DSpace：目标论文诱导的设计空间 → 前作落位 → 矩阵派生综合陈述 → 按维度写 related work。

Oracle 设定（给金标参考文献集，与 P6 的 B0/B1 同输入）。输出 StormParser 形态：
  p7/out/<variant>/<i>/storm_gen_article.md + url_to_info.json
其中 <i> = papers_with_related_works.csv 的行号（与 harness/eval 对齐）。

阶段：
  S1 设计空间诱导（LLM）：从目标论文标题+摘要抽 K 个"该文做出选择的设计维度"，每维给目标取值+前作可能取值。
  S2 前作落位（LLM，批量）：每篇被引论文在每维上的取值（或 n/a）+摘要依据短语+一句话做了什么。
  S3 矩阵派生（确定性，零 LLM）：每维按取值分组→类别陈述；与目标同值/异值→定位；目标取值格在前作中为空→缺口候选；
     全维与目标最近的前作→最近邻。
  S4 写作（LLM）：每维一段，段首类别级概括，段中分组引用，段尾目标定位；只许用 S3 给出的结构化事实做定位/缺口断言。
用法：python dspace.py --limit 5 --variant dspace_v1
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\review_1002\p6")
from llm27 import chat  # noqa: E402

csv.field_size_limit(10**9)
DSB = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/deepscholar/dsb/dataset/"
OUT = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/review_1002/p7/out/"


def load():
    rows = list(csv.DictReader(open(DSB + "papers_with_related_works.csv", encoding="utf-8")))
    ic = list(csv.DictReader(open(DSB + "important_citations.csv", encoding="utf-8")))
    by = {}
    for c in ic:
        by.setdefault(c["parent_paper_arxiv_id"], []).append(c)
    items = []
    for i, r in enumerate(rows):
        refs, seen = [], set()
        for c in by.get(r["arxiv_id"], []):
            t = (c["cited_paper_title"] or "").strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            refs.append({"n": len(refs) + 1, "title": t, "abstract": (c["cited_paper_abstract"] or "").strip()[:1500]})
        items.append({"i": i, "arxiv_id": r["arxiv_id"], "title": r["title"], "abstract": r["abstract"], "refs": refs})
    return items


def jparse(s):
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1)
    a, b = s.find("{"), s.rfind("}")
    return json.loads(s[a:b + 1])


S1 = """You are analyzing a research paper to understand how it positions itself against prior work.

Paper title: {title}
Abstract: {abstract}

Identify the {k} most important DESIGN DIMENSIONS along which this paper makes a deliberate choice that distinguishes it from prior work (e.g. "when control is applied: training-time vs inference-time", "degradation assumption: synthetic pre-defined vs real-world", "backbone: CNN / Transformer / state-space"). Order them by how central they are to the paper's contribution. For each dimension give: a short name, the value THIS paper takes, and 2-4 alternative values that prior work typically takes.

Return JSON only: {{"dimensions": [{{"name": "...", "target_value": "...", "alternatives": ["...", "..."], "why_it_matters": "one sentence"}}]}}"""

S2 = """Target paper: {title}
Design dimensions (with candidate values):
{dims}

For EACH prior paper below, place it on EACH dimension: pick the matching value (one of the listed values or the target value), write "other: <short value>" if it takes a different value, or "n/a" if the dimension does not apply / cannot be told from the text. Give a short evidence phrase copied from its abstract when you assign a value. Also give one sentence on what the paper does.

Prior papers:
{papers}

Return JSON only: {{"placements": [{{"n": <paper number>, "does": "...", "values": {{"<dimension name>": {{"value": "...", "evidence": "..."}}}}}}]}}"""

S4 = """Write the Related Work section for the paper below, in academic style, using inline numeric citations like [3] or [2, 5].

Paper title: {title}
Abstract: {abstract}

The section is organized by the paper's design dimensions. For each dimension write one paragraph (you may add a short subsection title): start with a category-level statement that characterizes how prior work handles this dimension (grouping papers that share a value and citing them together), contrast the groups, and end by stating where this paper stands relative to them. Use ONLY the structured facts and paper summaries below for claims about prior work; state gaps only where the facts mark a value as unoccupied among the cited works. Cover every cited paper at least once.

STRUCTURED FACTS (derived from placing each cited paper on each dimension):
{facts}

PAPER SUMMARIES:
{summaries}

Return only the Related Work text (markdown, starting with "## Related Work")."""


def s3_facts(dims, placements):
    facts = []
    for d in dims:
        name = d["name"]
        groups = {}
        for p in placements:
            v = ((p.get("values") or {}).get(name) or {}).get("value", "n/a")
            if not v or str(v).lower().startswith("n/a"):
                continue
            groups.setdefault(str(v).strip(), []).append(p["n"])
        tv = d.get("target_value", "")
        same = [n for v, ns in groups.items() if v.lower() == tv.lower() for n in ns]
        lines = [f"Dimension: {name} (why it matters: {d.get('why_it_matters', '')})",
                 f"  This paper's value: {tv}"]
        for v, ns in sorted(groups.items(), key=lambda x: -len(x[1])):
            lines.append(f"  Prior work with value '{v}': papers {sorted(ns)}")
        if not same:
            lines.append(f"  GAP: no cited prior paper takes this paper's value '{tv}' on this dimension.")
        else:
            lines.append(f"  Prior papers sharing this paper's value: {sorted(same)}")
        facts.append("\n".join(lines))
    # 最近邻：与目标同值维数最多的前作
    score = {}
    for p in placements:
        s = 0
        for d in dims:
            v = ((p.get("values") or {}).get(d["name"]) or {}).get("value", "")
            s += str(v).strip().lower() == d.get("target_value", "").strip().lower()
        score[p["n"]] = s
    if score:
        best = max(score.values())
        if best > 0:
            facts.append(f"Closest prior work (shares the most design choices with this paper, {best}/{len(dims)}): "
                         f"papers {sorted(n for n, s in score.items() if s == best)}")
    return "\n\n".join(facts)


def run_one(it, variant, k=4, batch=8):
    od = os.path.join(OUT, variant, str(it["i"]))
    if os.path.exists(os.path.join(od, "storm_gen_article.md")):
        return it["i"], "skip"
    os.makedirs(od, exist_ok=True)
    log = {"i": it["i"], "arxiv_id": it["arxiv_id"], "n_refs": len(it["refs"])}
    t0 = time.time()
    r1, _ = chat(S1.format(title=it["title"], abstract=it["abstract"], k=k), max_tokens=2500, temperature=0.2)
    dims = jparse(r1)["dimensions"][:k]
    dim_txt = "\n".join(f"- {d['name']}: this paper = {d['target_value']}; alternatives = {d.get('alternatives')}" for d in dims)
    placements = []
    for b in range(0, len(it["refs"]), batch):
        chunk = it["refs"][b:b + batch]
        ptxt = "\n\n".join(f"[{r['n']}] {r['title']}\n{r['abstract'] or '(no abstract available)'}" for r in chunk)
        for attempt in range(2):
            try:
                r2, _ = chat(S2.format(title=it["title"], dims=dim_txt, papers=ptxt), max_tokens=5000, temperature=0.1)
                placements += jparse(r2)["placements"]
                break
            except Exception as e:
                log.setdefault("s2_errors", []).append(str(e)[:120])
    facts = s3_facts(dims, placements)
    have = {p.get("n") for p in placements}
    summ = "\n".join(f"[{r['n']}] {r['title']}: " + next((p.get('does', '') for p in placements if p.get('n') == r['n']),
                                                       (r['abstract'][:300] or '(title only)'))
                     for r in it["refs"])
    r4, _ = chat(S4.format(title=it["title"], abstract=it["abstract"], facts=facts, summaries=summ),
                 max_tokens=4000, temperature=0.3)
    text = r4.strip()
    if not text.startswith("##"):
        text = "## Related Work\n\n" + text
    text += "\n\n## References\n\n" + "\n".join(f"[{r['n']}] {r['title']}" for r in it["refs"])
    open(os.path.join(od, "storm_gen_article.md"), "w", encoding="utf-8").write(text)
    json.dump({"url_to_unified_index": {f"ref://{r['n']}": r["n"] - 1 for r in it["refs"]},
               "url_to_info": {f"ref://{r['n']}": {"title": r["title"], "snippets": [r["abstract"][:500]]} for r in it["refs"]}},
              open(os.path.join(od, "url_to_info.json"), "w", encoding="utf-8"), ensure_ascii=False)
    log.update({"dims": dims, "n_placed": len(have), "facts": facts, "sec": round(time.time() - t0, 1)})
    json.dump(log, open(os.path.join(od, "trace.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return it["i"], f"ok {log['sec']}s dims={len(dims)} placed={len(have)}/{len(it['refs'])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--variant", default="dspace_v1")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    items = load()[a.offset:]
    if a.limit:
        items = items[:a.limit]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(run_one, it, a.variant) for it in items]):
            try:
                print(*fut.result(), flush=True)
            except Exception as e:
                print("ERR", str(e)[:200], flush=True)


if __name__ == "__main__":
    main()
