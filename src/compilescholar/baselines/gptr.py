# -*- coding: utf-8 -*-
"""GPT-Researcher arm: report -> CS2 sections adapter (moved verbatim from cs2/cs2_gptr.py).

Only the deterministic adapter lives here. The runner (GPT-Researcher + local 27B + Sciverse through the old tiered
SearchService path) stays in legacy/ unchanged, because rewiring its retrieval to compilescholar.sources would change
the arm's behaviour; whether to re-run it on the unified search path is decided by W1-8 (empty-retrieval rate).
Reported GPTR numbers come from that legacy runner.

`_SNIPPETS` (url -> retrieved abstract) is filled by the retriever during a run and used as citation snippets.
"""
from __future__ import annotations

import re

_NUM_CITE = re.compile(r"\[(\d+)\]")
# 第 10 坑（09-30）：v0.16 report prompt 用 APA markdown 超链接引用
# （([in-text citation](url)) 句尾形态），不是 [1] 编号——旧 parser 只认
# 编号 → cites=0。md 链接正则（普通链接+括号包裹链接两种形态）。
_MD_CITE = re.compile(r"\(?\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)?\)?")
_MD_CITE_PAREN = re.compile(r"\(\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)\)")


# url -> 检索返回的 raw_content（标题+摘要）；SciverseRetriever.search 登记，
# report_to_cs2 用作 snippet。进程内共享（同一运行内 url 唯一对应一篇论文）。
_SNIPPETS: dict = {}


def report_to_cs2(report_md: str, sources=None) -> list[dict]:
    """GPT Researcher 报告 → CS2 sections/citations。两种引用形态：
    ① 正文 [1][2] 编号 + 文末 sources 列表（v0.15-）
    ② v0.16 APA markdown 超链接（([in-text citation](url)) 句尾）
    url 是稳定键：同 url 多次引用共用一条，title 取首次出现的链接文字。"""
    # 形态②：markdown 链接直接建 url→citation 映射
    url_map = {}   # url -> {"title":..., "n": seq}
    for m in _MD_CITE_PAREN.finditer(report_md):
        t, u = m.group(1).strip(), m.group(2)
        if u not in url_map and t and not t.lower().startswith("http"):
            url_map[u] = {"title": t[:120]}
    for m in re.finditer(r"\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)",
                         report_md):
        t, u = m.group(1).strip(), m.group(2)
        if u not in url_map and t and not t.lower().startswith("http"):
            url_map[u] = {"title": t[:120]}
    # 形态①：正文编号 + 文末 sources 列表（行首锚定——正文句中 [1]
    # 不是 sources 行）
    src_map = {}
    for m in re.finditer(r"^\s*\[(\d+)\][\s\-–]*(.+?)$",
                         report_md, re.M):
        n, rest = int(m.group(1)), m.group(2).strip()
        if n in src_map or not rest:
            continue
        title = re.sub(r"\[|\]\(.*?\)|https?://\S+", "", rest).strip()[:120]
        url_m = re.search(r"(https?://\S+)", rest)
        src_map[n] = {"title": title or f"source_{n}",
                      "url": url_m.group(1) if url_m else ""}
    # 分节
    sections = []
    cur_title, cur_lines = "Answer", []
    for line in report_md.splitlines():
        if line.startswith("## "):
            if any(l.strip() for l in cur_lines):
                sections.append((cur_title, "\n".join(cur_lines)))
            cur_title, cur_lines = line[3:].strip() or "Section", []
        else:
            cur_lines.append(line)
    if any(l.strip() for l in cur_lines):
        sections.append((cur_title, "\n".join(cur_lines)))

    out = []
    for title, text in sections:
        if title.lower() in ("references", "reference", "sources",
                             "table of contents"):
            continue
        if not text.strip():
            continue
        cites, seen = [], {}
        # 形态②：markdown 超链接引用（v0.16）——url 为键，节内去重，
        # 重编号 [1]..[n]；链接文字保留在正文（原样不动，judges 可见）
        for m in _MD_CITE.finditer(text):
            u = m.group(2)
            if u in seen:
                continue
            info = url_map.get(u) or {"title": u[:100]}
            if u not in url_map:
                url_map[u] = info
            seen[u] = len(cites) + 1
            cites.append({
                "id": f"[{seen[u]}]",
                # 有检索到的摘要就作 snippet（与 ours 的 abstract 档对称；
                # REBUILD-PLAN-1003 E5）——retriever 契约里本就带 raw_content
                "snippets": ([_SNIPPETS[u][:1600]] if u in _SNIPPETS else []),
                "title": info["title"],
                "metadata": {"url": u},
            })
        # 形态①：[n] 编号引用
        for m in _NUM_CITE.finditer(text):
            n = int(m.group(1))
            if ("md", n) in seen or n not in src_map:
                continue
            seen[("md", n)] = len(cites) + 1
            info = src_map[n]
            cites.append({
                "id": f"[{seen[('md', n)]}]",
                # GPT Researcher 无 snippet 抽取——title 档（0.5）；
                # raw_content 可作 snippet（retriever 契约里有全文）
                "snippets": [],
                "title": info["title"],
                "metadata": {"url": info["url"]},
            })
        text2 = _NUM_CITE.sub(
            lambda m: f"[{seen[('md', int(m.group(1)))]}]"
            if ("md", int(m.group(1))) in seen else m.group(0), text)
        # 官方契约：citation id 必须原样出现在正文（task.py:81）。之前 id 改成 [k]
        # 但正文仍是 md 链接 → grader 返回的 supporting 是标题文本、与 id 永不相交 →
        # 半信用扣减静默失效（REBUILD-PLAN-1003 E5）。把正文里的 md 链接替换成 [k]：
        # 先替换外层带括号的形态，再替换裸链接。
        def _md_to_id(m):
            k = seen.get(m.group(2))
            return f"[{k}]" if k else m.group(0)
        text2 = _MD_CITE_PAREN.sub(_md_to_id, text2)
        text2 = re.sub(r"\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)", _md_to_id, text2)
        out.append({"title": title, "text": text2, "citations": cites})
    return out
