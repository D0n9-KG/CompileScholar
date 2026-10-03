"""对照臂 TopicCluster：与 DSpace 同预算（同 27B、同 oracle 输入、同四阶段调用量级），
唯一差别=组织轴来自被引论文之间的主题聚类（STORM/GRASP/HiGTL 族的做法），而非目标论文诱导的设计维度。
  T1（LLM）：只看被引论文标题+摘要，把它们分成 4 个主题组并命名（不看目标论文）。
  T2（LLM，批量）：每篇写一句"做了什么"。
  T3（确定性）：组→成员列表。
  T4（LLM）：每组一段写 related work（段首类别概括+分组引用+段尾与目标论文的关系），与 DSpace S4 同模板强度。
输出与 DSpace 相同：p6/gen/TopicCluster/<gt_dir>.md
"""
import concurrent.futures as cf
import json
import os
import re
import sys
import time

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\review_1002\p6")
from llm27 import chat  # noqa: E402

P6 = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/review_1002/p6/"
OUTD = P6 + "gen/TopicCluster/"


def jparse(s):
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)
    s = m.group(1) if m else s
    a, b = s.find("{"), s.rfind("}")
    return json.loads(s[a:b + 1])


T1 = """Below are papers cited in a related-work section. Group them into {k} coherent TOPIC groups based on what the papers are about (their research topics), and give each group a short descriptive name. Every paper must belong to exactly one group.

{papers}

Return JSON only: {{"groups": [{{"name": "...", "papers": [<paper numbers>]}}]}}"""

T2 = """For each paper below, write one sentence on what it does.

{papers}

Return JSON only: {{"summaries": [{{"n": <paper number>, "does": "..."}}]}}"""

T4 = """Write the Related Work section for the paper below, in academic style, using inline numeric citations like [3] or [2, 5].

Paper title: {title}
Abstract: {abstract}

The section is organized by topic groups of the cited works. For each group write one paragraph (you may add a short subsection title): start with a category-level statement that characterizes the group (citing its papers together), contrast approaches within it, and end by stating how this paper relates to that line of work. Cover every cited paper at least once.

TOPIC GROUPS:
{groups}

PAPER SUMMARIES:
{summaries}

Return only the Related Work text (markdown, starting with "## Related Work")."""


def run(e, k=4, batch=8):
    out = OUTD + f"{e['gt_dir']}.md"
    if os.path.exists(out):
        return e["gt_dir"], "skip"
    t0 = time.time()
    refs = [{"n": j + 1, **r} for j, r in enumerate(e["refs"])]
    ptxt_all = "\n\n".join(f"[{r['n']}] {r['title']}\n{(r.get('abstract') or '(no abstract available)')[:1500]}" for r in refs)
    g = jparse(chat(T1.format(k=k, papers=ptxt_all), max_tokens=2500, temperature=0.2)[0])["groups"][:k]
    summ = {}
    for b in range(0, len(refs), batch):
        chunk = refs[b:b + batch]
        ptxt = "\n\n".join(f"[{r['n']}] {r['title']}\n{(r.get('abstract') or '(no abstract available)')[:1500]}" for r in chunk)
        try:
            for s in jparse(chat(T2.format(papers=ptxt), max_tokens=3000, temperature=0.1)[0])["summaries"]:
                summ[s.get("n")] = s.get("does", "")
        except Exception:
            pass
    gtxt = "\n".join(f"- {x['name']}: papers {sorted(x['papers'])}" for x in g)
    stxt = "\n".join(f"[{r['n']}] {r['title']}: " + (summ.get(r['n']) or (r.get('abstract') or '')[:300] or '(title only)') for r in refs)
    text = chat(T4.format(title=e["title"], abstract=e["abstract"], groups=gtxt, summaries=stxt),
                max_tokens=4000, temperature=0.3)[0].strip()
    open(out, "w", encoding="utf-8").write(text)
    return e["gt_dir"], f"ok {time.time() - t0:.0f}s groups={len(g)}"


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    oi = json.load(open(P6 + "oracle_inputs.json", encoding="utf-8"))
    with cf.ThreadPoolExecutor(5) as ex:
        for f in cf.as_completed([ex.submit(run, e) for e in oi]):
            try:
                print(*f.result(), flush=True)
            except Exception as ex_:
                print("ERR", str(ex_)[:200], flush=True)
