# -*- coding: utf-8 -*-
"""Memorized baselines -> official CS2 format adapter + GLM re-judging prep.

Three baselines (from allenai/asta-bench-solver-data):
  Elicit      responses.json  — already in sections format (direct)
  Perplexity  openai_dr/*.json — markdown report (needs sections wrapping)
  SciSpace    scispace/test.json — response text (needs sections wrapping)

The wrapping follows the official format_solver convention (single
section, no citations — these systems don't emit citation JSON;
citation facets score via the is_retrieverless 0.5-title-tier path,
disclosed).

Output: arm_memorized/answers_{system}_{split}.json — rows keyed by
question text matching the rubrics, ready for the official scorer.
"""
import json
import re
import sys
from pathlib import Path

from ..core import paths

CS2 = paths.legacy_bench() / "cs2"
MEM = CS2 / "arm_memorized"
RUBRICS = CS2.parent / "scholarqa_multi" / "sqa2_rubrics_v1_recomputed.json"


def load_rubrics(split="dev"):
    version = "v1" if split == "dev" else "v2"
    p = CS2.parent / "scholarqa_multi" / f"sqa2_rubrics_{version}_recomputed.json"
    return json.load(open(p, encoding="utf-8"))


def adapt_elicit(split="dev"):
    """Elicit responses.json: rows are {id, sections} — id is a topic
    slug, not the case_id. Match to rubrics by ... we can't match by id;
    the dataset covers both splits (201 rows). For dev judging we need
    the question mapping — Elicit rows are ordered by their original
    evaluation; absent a key, match by content overlap is unreliable.
    PRACTICAL route: judge the FULL Elicit set against whichever rubric
    question the sections answer — the official scorer needs
    question+response pairs. We emit Elicit rows paired with their
    best-matching rubric question via title-token overlap.
    """
    # 10-03 修正：存档 id 就是题面原文（201 行里 200 行归一化后与 dev/test 题面逐字相等，一一对应）。
    # 旧版按 token 重叠"最佳匹配"配题——148 行配到 100 个 dev qid，其中 48 行是别的题的回答被配给 dev 题
    # （同一 qid 多行时后写覆盖前写），判分输入错位。改为归一化精确匹配，无匹配则不配。
    data = json.load(open(MEM / "sqa_elicit_responses.json",
                          encoding="utf-8"))
    rubrics = load_rubrics(split)
    norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()  # noqa: E731
    qmap = {norm(q["question"]): q for q in rubrics}
    out = []
    for r in data:
        q = qmap.get(norm(r.get("id")))
        if q is None:
            continue
        # 10-03：Elicit 存档的论文标题在 citation.metadata.paper.title，顶层没有 title；官方 scorer 在片段与正文重合时
        # 走"只给标题"分支读 c["title"] → KeyError，整题引用两项被记 0（test 10/100 题）。按官方 sections 形态把标题提到顶层。
        secs = []
        for s in r.get("sections") or []:
            cites = []
            for c in s.get("citations") or []:
                c = dict(c)
                if not c.get("title"):
                    c["title"] = ((c.get("metadata") or {}).get("paper") or {}).get("title") or ""
                cites.append(c)
            secs.append({**s, "citations": cites})
        out.append({"qid": q["case_id"][:24],
                    "question": q["question"],
                    "sections": secs,
                    "source": "elicit"})
    return out


def adapt_text_system(name, fname, split, answer_field):
    """Perplexity/SciSpace: {question, answer/response} rows — match
    question text to rubrics (exact/normalized)."""
    data = json.load(open(MEM / fname, encoding="utf-8"))
    rubrics = load_rubrics(split)
    qmap = {}
    for q in rubrics:
        qmap[_norm(q["question"])] = q
    out = []
    for r in data:
        q = qmap.get(_norm(r.get("question") or ""))
        if not q:
            continue
        text = r.get(answer_field) or ""
        if not text:
            continue
        # markdown report -> single section (official format_solver
        # convention: text systems emit one body section, no citations)
        out.append({"qid": q["case_id"][:24],
                    "question": q["question"],
                    "sections": [{"title": "Report", "text": text,
                                  "citations": []}],
                    "source": name})
    return out


def _norm(s):
    import re
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def adapt_openai_dr(fname, split):
    """OpenAI Deep Research 存档（asta-bench-solver-data/sqa/openai_dr）→ CS2 sections。

    之前此文件被误标为 "perplexity_dr" 且 citations=[] 写死（REBUILD-PLAN-1003 E4）。
    存档格式：正文用 [n] 内联引用；文末 "\\nCitations:\\n" 块逐条 "[n] 标题- url: 摘录"。
    这里把摘录作 snippet、标题作 title，正文保留 [n]（id 出现在正文，满足 task.py:81）。
    正文按 markdown "## " 分节；每节只挂本节实际出现的 [n]。"""
    import re
    data = json.load(open(MEM / fname, encoding="utf-8"))
    qmap = {_norm(q["question"]): q for q in load_rubrics(split)}
    entry = re.compile(r"^\[(\d+)\]\s*(.*?)(?=^\[\d+\]|\Z)", re.S | re.M)
    out = []
    for r in data:
        q = qmap.get(_norm(r.get("question") or ""))
        text = r.get("answer") or ""
        if not q or not text:
            continue
        body, _, cblock = text.partition("\nCitations:")
        refs = {}
        for m in entry.finditer(cblock.strip()):
            n, rest = int(m.group(1)), m.group(2).strip()
            head, _, snip = rest.partition("\n")
            title = re.split(r"-\s*(?:https?://|$)", head, maxsplit=1)[0].strip(" -")
            um = re.search(r"https?://\S+?(?=:\s|$)", head)
            if not snip.strip():  # "标题- url: 摘录" 同行形态
                snip = head.split(": ", 1)[1] if ": " in head else ""
            refs[n] = {"title": title[:200] or f"source {n}",
                       "url": um.group(0) if um else "",
                       "snippet": re.sub(r"\s+", " ", snip.lstrip("> ")).strip()[:1600]}
        parts = re.split(r"(?m)^(#{1,3} .+)$", body)
        secs, cur_t, cur = [], "Report", parts[0]
        for i in range(1, len(parts), 2):
            if cur.strip():
                secs.append((cur_t, cur))
            cur_t, cur = parts[i].lstrip("# ").strip(), parts[i + 1] if i + 1 < len(parts) else ""
        if cur.strip():
            secs.append((cur_t, cur))
        sections = []
        for t, s in secs:
            ns = sorted({int(x) for g in re.findall(r"\[([\d,\s–-]+)\]", s)
                         for x in re.findall(r"\d+", g) if int(x) in refs})
            cites = [{"id": f"[{n}]", "snippets": [refs[n]["snippet"]] if refs[n]["snippet"] else [],
                      "title": refs[n]["title"], "metadata": {"url": refs[n]["url"]}} for n in ns]
            sections.append({"title": t[:200], "text": s.strip(), "citations": cites})
        out.append({"qid": q["case_id"][:24], "question": q["question"],
                    "sections": sections, "source": "openai_dr"})
    return out


def adapt_scispace(split="test"):
    """SciSpace 存档（asta-bench-solver-data/sqa/scispace/test.json，只有 test）：{question, response:{sections}}，
    已是官方 sections 形态（citation id 在正文 4244/4245、全部带 snippets）。只做题面精确配题 + 保留 title/snippets/id
    （paper 元数据收进 metadata），不改正文。"""
    data = json.load(open(MEM / "sqa_scispace_test.json", encoding="utf-8"))
    norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()  # noqa: E731
    qmap = {norm(q["question"]): q for q in load_rubrics(split)}
    out = []
    for r in data:
        q = qmap.get(norm(r.get("question")))
        if q is None:
            continue
        secs = []
        for s in (r.get("response") or {}).get("sections") or []:
            cites = [{"id": c.get("id"), "snippets": c.get("snippets") or [],
                      "title": (c.get("paper") or {}).get("title"),
                      "metadata": {"year": (c.get("paper") or {}).get("year"), "corpus_id": c.get("corpus_id")}}
                     for c in s.get("citations") or []]
            secs.append({"title": s.get("title") or "", "text": s.get("text") or "", "citations": cites})
        out.append({"qid": q["case_id"][:24], "question": q["question"], "sections": secs, "source": "scispace"})
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = []
    # OpenAI Deep Research dev（此文件之前被误标 perplexity_dr，见 adapt_openai_dr 文档）
    rows = adapt_openai_dr("sqa_openai_dr_dev.json", "dev")
    n_c = sum(len(s["citations"]) for r in rows for s in r["sections"])
    print(f"openai_dr dev matched: {len(rows)}/100 | citations {n_c}")
    json.dump(rows, open(MEM / "answers_openai_dr_dev.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    # SciSpace 只有 test——dev20 用不了，记录
    # Elicit
    rows = adapt_elicit("dev")
    print(f"elicit dev matched: {len(rows)}")
    json.dump(rows, open(MEM / "answers_elicit_dev.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
