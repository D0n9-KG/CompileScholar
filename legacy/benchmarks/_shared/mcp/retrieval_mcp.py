# -*- coding: utf-8 -*-
"""Phase 0.3 检索/语料 MCP 服务器（槽 3/4/5 harness 臂共用）。

EXPERIMENT-DESIGN-WORLDSTATE-0926 §六 0.3：search_text/fetch_chunk/
Sciverse 包装。消费方=Claude-Code 级 harness 臂（槽 3/4 开网臂、槽 5
固定语料断网臂）；我方 broker 走内部工具不经此层。

双模式（env RETRIEVAL_MCP_MODE）：
  open   槽 3/4 —— Sciverse 语义检索（sciverse-semantic 全域打头，
         A6 判决的词汇鸿沟主杠杆）+ /content 原文分块读取；before_year
         参数承载 CS2 2025-05-01 截止纪律（服务器侧过滤，防臂侧遗忘）
  corpus 槽 5 —— 固定语料断网：manifest+texts 本地索引（BM25 式词法
         评分，零外部依赖零 GPU），fetch_chunk 读本地分块

两种模式工具面一致（search_papers/fetch_chunk/list_corpus），harness
prompt 与模式无关；open 模式 list_corpus 返回开放语料说明。

接线（Claude Code）：
  claude -p --mcp-config '{"mcpServers":{"retrieval":{\
    "command":"python","args":["<本文件>"]}}}' \
    --allowedTools "mcp__retrieval__search_papers,mcp__retrieval__fetch_chunk"

env：
  RETRIEVAL_MCP_MODE=open|corpus（默认 open）
  RETRIEVAL_MCP_MANIFEST / RETRIEVAL_MCP_TEXTS（corpus 模式必填）
  SCIVERSE_API_TOKEN（open 模式；缺省从 CompileScholar/.env 读）
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

CS_REPO = Path(r"C:\Users\D0n9\Desktop\CompileScholar")
for _p in (str(CS_REPO / "src"),
           str(Path(__file__).resolve().parent.parent / "tools")):  # cutoff.py
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcp.server.fastmcp import FastMCP  # noqa: E402

MODE = os.environ.get("RETRIEVAL_MCP_MODE", "open").lower()
mcp = FastMCP("retrieval")

_CHUNK_CHARS = 1200


def _load_env_token() -> str | None:
    """CompileScholar/.env 的 SCIVERSE_API_TOKEN（k2 分仓纪律：本仓 key
    归答题侧检索；sci-evo-extract/.env 的 k1 归建库侧，此处不读）。"""
    if os.environ.get("SCIVERSE_API_TOKEN"):
        return os.environ["SCIVERSE_API_TOKEN"]
    for line in (CS_REPO / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*SCIVERSE_API_TOKEN\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    return None


# ================================================================ corpus 模式

class CorpusIndex:
    """固定语料的词法索引：manifest+texts → 论文/分块两级检索。

    评分=查询词 tf-idf 加和（标题命中×3 加权）；零外部服务依赖——
    槽 5 断网纪律决定了不能用嵌入/外部 API。"""
    _TOK = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")

    def __init__(self, manifest_path: str, texts_dir: str):
        self.manifest = json.load(open(manifest_path, encoding="utf-8"))
        self.texts_dir = Path(texts_dir)
        self.papers: dict[str, dict] = {r["paper_id"]: r for r in self.manifest}
        self.chunks: list[dict] = []      # {paper_id, chunk_id, text}
        self._df: dict[str, int] = {}
        for pid in self.papers:
            path = self.texts_dir / f"{pid}.md"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for i in range(0, len(text), _CHUNK_CHARS):
                self.chunks.append({
                    "paper_id": pid,
                    "chunk_id": f"{pid}#c{i // _CHUNK_CHARS}",
                    "text": text[i:i + _CHUNK_CHARS],
                })
        for c in self.chunks:
            for t in set(self._tok(c["text"])):
                self._df[t] = self._df.get(t, 0) + 1
        self.n_docs = max(1, len(self.chunks))

    def _tok(self, s: str) -> list[str]:
        return self._TOK.findall(s.lower())

    def _score(self, qtoks: list[str], text: str, boost: float = 1.0) -> float:
        toks = self._tok(text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in qtoks:
            if q in tf:
                idf = math.log(1 + self.n_docs / (1 + self._df.get(q, 0)))
                s += math.log(1 + tf[q]) * idf
        return s * boost

    def search(self, query: str, k: int) -> list[dict]:
        qtoks = self._tok(query)
        if not qtoks:
            return []
        scored = []
        for c in self.chunks:
            s = self._score(qtoks, c["text"])
            title = self.papers.get(c["paper_id"], {}).get("title") or ""
            s += self._score(qtoks, title, boost=3.0) * 0.3
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        # 论文级去重：同篇多块只留最高分块（返回论文+最佳块）
        seen: set[str] = set()
        out = []
        for s, c in scored:
            if c["paper_id"] in seen:
                continue
            seen.add(c["paper_id"])
            meta = self.papers.get(c["paper_id"], {})
            out.append({
                "paper_id": c["paper_id"],
                "title": meta.get("title"),
                "year": meta.get("year"),
                "doi": meta.get("doi"),
                "chunk_id": c["chunk_id"],
                "matched_text": c["text"][:400],
                "score": round(s, 3),
            })
            if len(out) >= k:
                break
        return out

    def fetch(self, paper_id: str, chunk_id: str | None,
              offset: int | None, limit: int) -> dict:
        path = self.texts_dir / f"{paper_id}.md"
        if not path.exists():
            return {"error": f"unknown paper_id: {paper_id}"}
        text = path.read_text(encoding="utf-8", errors="replace")
        if chunk_id:
            m = re.search(r"#c(\d+)$", chunk_id)
            if not m:
                return {"error": f"bad chunk_id: {chunk_id}"}
            i = int(m.group(1)) * _CHUNK_CHARS
            return {"paper_id": paper_id, "chunk_id": chunk_id,
                    "text": text[i:i + _CHUNK_CHARS]}
        start = (offset or 0) * _CHUNK_CHARS
        return {
            "paper_id": paper_id,
            "chunks": [
                {"chunk_id": f"{paper_id}#c{(start + j) // _CHUNK_CHARS}",
                 "text": text[start + j:start + j + _CHUNK_CHARS]}
                for j in range(0, min(limit, 8) * _CHUNK_CHARS,
                               _CHUNK_CHARS)
            ],
            "total_chars": len(text),
        }


_CORPUS: CorpusIndex | None = None


def _corpus() -> CorpusIndex:
    global _CORPUS
    if _CORPUS is None:
        man = os.environ.get("RETRIEVAL_MCP_MANIFEST")
        txt = os.environ.get("RETRIEVAL_MCP_TEXTS")
        if not man or not txt:
            raise RuntimeError(
                "corpus mode requires RETRIEVAL_MCP_MANIFEST + "
                "RETRIEVAL_MCP_TEXTS env")
        _CORPUS = CorpusIndex(man, txt)
    return _CORPUS


# ================================================================ open 模式

def _sciverse():
    from sci_evo_extract.library.sources import SciverseClient
    # 坑（亲测）：sciverse_request_json 只读环境变量 SCIVERSE_API_TOKEN，
    # SciverseClient(token=...) 构造参数被忽略——必须先设 env 再构造。
    tok = _load_env_token()
    if tok:
        os.environ["SCIVERSE_API_TOKEN"] = tok
    return SciverseClient(token=tok)


def _open_search(query: str, k: int, before_year: int | None) -> list[dict]:
    client = _sciverse()
    from cutoff import cutoff as _c0
    # 截止开启时语义检索头部多为 2025+ 新文（实测 16 取 4），多取以保证 k 条有效
    cands = client.semantic_search(query, limit=max(k * (5 if _c0() else 2), 12))
    out = []
    from cutoff import allowed as _allowed, cutoff as _cut
    for c in cands:
        if c.status != "ready":
            continue
        year = c.year
        # 知识截止（REBUILD-PLAN-1003 E3）：服务端强制，不依赖模型是否传 before_year
        # （harness 691 次检索 0 次传参）；与 external_tools 同一实现（_shared/tools/cutoff.py）
        if _cut() is not None and not _allowed(year):
            continue
        if before_year is not None and year is not None and year > before_year:
            continue
        raw = c.raw or {}
        out.append({
            "doc_id": raw.get("doc_id"),
            "title": c.title,
            "year": year,
            "venue": c.venue,
            "matched_text": (raw.get("chunk") or raw.get("abstract") or "")[:400],
            "chunk_id": raw.get("chunk_id"),
            "score": c.candidate_score,
        })
        if len(out) >= k:
            break
    return out


# ================================================================ 工具面

@mcp.tool()
def search_papers(query: str, k: int = 8, before_year: int | None = None) -> str:
    """Search the paper corpus for a research query.

    Returns up to k papers, each with title/year/venue and the matched
    text passage (a verbatim excerpt from the paper). Use before_year
    (e.g. 2025) to restrict to papers published in or before that year.

    Args:
        query: research query (natural language or keywords)
        k: max papers to return (default 8)
        before_year: only papers published <= this year (optional)
    """
    try:
        if MODE == "corpus":
            rows = _corpus().search(query, k)
        else:
            rows = _open_search(query, k, before_year)
        return json.dumps(
            {"mode": MODE, "n": len(rows), "papers": rows},
            ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "mode": MODE}, ensure_ascii=False)


_FETCH_TEXT_CAP = 12000  # 单次返回正文上限（防 64KB 全文撑爆 harness 上下文）


@mcp.tool()
def fetch_chunk(doc_id: str, chunk_id: str | None = None,
                offset: int | None = None, limit: int = 4) -> str:
    """Read the full text of a paper found via search_papers.

    In open mode, doc_id is the Sciverse doc_id from search results; passing
    the chunk_id from search results returns the full document text. In
    corpus mode, doc_id is the paper_id; use chunk_id or offset+limit for
    sequential chunks. Long texts are truncated with a total_chars marker.

    Args:
        doc_id: document id from search_papers results
        chunk_id: specific chunk id (optional, from search results)
        offset: chunk offset to start reading from (optional)
        limit: number of sequential chunks (default 4)
    """
    try:
        if MODE == "corpus":
            data = _corpus().fetch(doc_id, chunk_id, offset, limit)
        else:
            client = _sciverse()
            raw = client.read_content(
                doc_id=doc_id, chunk_id=chunk_id,
                offset=offset if chunk_id is None else None,
                limit=limit if chunk_id is None else None,
            )
            text = str(raw.get("text") or "")
            data = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "total_chars": len(text),
                "truncated": len(text) > _FETCH_TEXT_CAP,
                "text": text[:_FETCH_TEXT_CAP],
            }
        return json.dumps(data, ensure_ascii=False)[:14000]
    except Exception as e:
        return json.dumps({"error": str(e), "doc_id": doc_id},
                          ensure_ascii=False)


@mcp.tool()
def list_corpus() -> str:
    """List the papers available for search (corpus mode only)."""
    if MODE == "corpus":
        c = _corpus()
        rows = [{"paper_id": pid, "title": r.get("title"),
                 "year": r.get("year"), "doi": r.get("doi")}
                for pid, r in c.papers.items()]
        return json.dumps({"mode": "corpus", "n": len(rows), "papers": rows},
                          ensure_ascii=False)[:12000]
    return json.dumps({
        "mode": "open",
        "note": ("open-world retrieval: search_papers reaches the full "
                 "literature index; use before_year to enforce a knowledge "
                 "cutoff")}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
