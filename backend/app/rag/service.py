from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from app.graph.neo4j_client import Neo4jClient
from app.ingest.paper_meta import load_canonical_meta
from app.rag.retrieval import latest_run_dir, load_chunks_from_run, lexical_retrieve
from app.settings import settings
from app.vector.faiss_store import load_faiss

log = logging.getLogger(__name__)


def _runs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs"


def _storage_dir() -> Path:
    p = Path(__file__).resolve().parents[2] / settings.storage_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def global_faiss_dir() -> Path:
    return _storage_dir() / "faiss"


def latest_faiss_dir() -> Path:
    g = global_faiss_dir()
    if g.exists():
        return g
    latest = _runs_dir() / "LATEST"
    if not latest.exists():
        raise FileNotFoundError("No FAISS index yet. Build one via /tasks/rebuild/faiss or call /ingest/path first.")
    run_id = latest.read_text(encoding="utf-8").strip()
    faiss_dir = _runs_dir() / run_id / "faiss"
    if not faiss_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found for run {run_id}. Build global index via /tasks/rebuild/faiss or re-run ingest with an embeddings-capable provider."
        )
    return faiss_dir


def _paper_id_from_md_path(md_path: str | None) -> str | None:
    if not md_path:
        return None
    meta = load_canonical_meta(md_path)
    doi = str(meta.get("doi") or "").strip().lower()
    if not doi:
        return None
    return f"doi:{doi}"

def _allowed_paper_sources(scope: dict | None) -> set[str] | None:
    if not scope:
        return None
    mode = str(scope.get("mode") or "all").strip().lower()
    if mode == "all":
        return None
    if mode == "collection":
        cid = str(scope.get("collection_id") or "").strip()
        if not cid:
            return set()
        try:
            with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
                return set(client.list_paper_sources_for_collection(cid))
        except Exception:
            return set()
    if mode == "papers":
        ids = scope.get("paper_ids") or []
        paper_ids = [str(x).strip() for x in ids if str(x).strip()]
        if not paper_ids:
            return set()
        try:
            with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
                return set(client.list_paper_sources_for_paper_ids(paper_ids))
        except Exception:
            return set()
    return None


# ---------------------------------------------------------------------------
# Hybrid retrieval helpers
# ---------------------------------------------------------------------------

def _rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]],
    k_rrf: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked result lists.

    Each item must have a ``chunk_id`` key.  Returns a single list sorted by
    fused score (descending).  Duplicate chunk_ids within the same list are
    ignored; across lists they are merged.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        seen_in_ranked: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            if not isinstance(item, dict):
                continue
            raw_cid = item.get("chunk_id")
            cid = str(raw_cid).strip() if raw_cid is not None else ""
            if not cid or cid.lower() == "none" or cid in seen_in_ranked:
                continue
            seen_in_ranked.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
            if cid not in first_seen:
                normalized = dict(item)
                normalized["chunk_id"] = cid
                first_seen[cid] = normalized
    fused = []
    for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(first_seen[cid])
        entry["rrf_score"] = score
        fused.append(entry)
    return fused


def _build_system_prompt(domain_prompt: str | None = None) -> str:
    """Build the RAG system prompt with configurable domain context."""
    domain = (domain_prompt or "").strip()
    if not domain:
        domain = "You are a scientific research assistant."
    return (
        f"{domain}\n"
        "Answer ONLY using the provided evidence snippets, validated claims, and graph context.\n"
        "If evidence is insufficient, say what is missing.\n"
        "Cite evidence by referencing the evidence ids like [E1], [E2].\n"
        "When referencing validated claims, use their claim id like [CL:abc123].\n"
        "When graph context is provided, use it to enrich your answer with "
        "structural relationships (citations, logic steps, claims)."
    )


def _stringify_graph_value(value: Any, *, max_chars: int = 240) -> str:
    """Convert a graph context value to a compact string."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = ", ".join(str(v).strip() for v in value if str(v).strip())
    elif isinstance(value, dict):
        text = ", ".join(
            f"{k}={v}" for k, v in value.items() if str(k).strip() and str(v).strip()
        )
    else:
        text = str(value).strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _format_graph_context(graph_context: list[dict[str, Any]] | None) -> str:
    """Format graph context entries into a text block for the LLM prompt.

    Caps at 30 entries and 6000 total characters to avoid token overflow.
    Supports list/dict values from Neo4j citation context.
    """
    if not graph_context:
        return ""
    field_order = (
        "paper_source", "doi", "cited_doi", "cited_title", "purpose_labels",
        "total_mentions", "ref_nums", "source_paper", "target_paper",
        "relationship", "purpose", "step_type", "summary",
    )
    max_entries = 30
    max_total_chars = 6000
    header = "Graph Context:"
    remaining = max_total_chars - len(header) - 1
    lines: list[str] = []
    for entry in graph_context[:max_entries]:
        if remaining <= 0:
            break
        if not isinstance(entry, dict):
            continue
        parts = []
        for key in field_order:
            val = _stringify_graph_value(entry.get(key))
            if val:
                parts.append(f"{key}={val}")
        if not parts:
            continue
        line = " | ".join(parts)
        if len(line) > remaining:
            cutoff = max(0, remaining - 3)
            line = (line[:cutoff].rstrip() + "...") if cutoff else ""
        if not line:
            break
        lines.append(line)
        remaining -= len(line) + 1
    if not lines:
        return ""
    return header + "\n" + "\n".join(lines)


def _format_structured_knowledge(knowledge: dict[str, list[dict[str, Any]]] | None) -> str:
    """Format claims and logic steps into a text block for the LLM prompt.

    Each claim includes its claim_id so the LLM can reference it in the answer,
    enabling frontend traceability (e.g. [CL:abc123]).
    """
    if not knowledge:
        return ""
    parts: list[str] = []

    # Logic steps
    steps = knowledge.get("logic_steps") or []
    if steps:
        step_lines = []
        for s in steps[:20]:
            st = str(s.get("step_type") or "").strip()
            summary = str(s.get("summary") or "").strip()
            ps = str(s.get("paper_source") or "").strip()
            if st and summary:
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                step_lines.append(f"  [{ps}] {st}: {summary}")
        if step_lines:
            parts.append("Logic Steps:\n" + "\n".join(step_lines))

    # Claims
    claims = knowledge.get("claims") or []
    if claims:
        claim_lines = []
        for c in claims[:30]:
            cid = str(c.get("claim_id") or "").strip()
            text = str(c.get("text") or "").strip()
            st = str(c.get("step_type") or "").strip()
            conf = c.get("confidence")
            ps = str(c.get("paper_source") or "").strip()
            if cid and text:
                if len(text) > 300:
                    text = text[:297] + "..."
                conf_str = (
                    f" (conf={conf:.2f})"
                    if isinstance(conf, (int, float)) and not isinstance(conf, bool)
                    else ""
                )
                scope = "/".join(part for part in (ps, st) if part)
                scope_str = f" [{scope}]" if scope else ""
                claim_lines.append(f"  [CL:{cid}]{scope_str}{conf_str} {text}")
        if claim_lines:
            parts.append("Validated Claims:\n" + "\n".join(claim_lines))

    if not parts:
        return ""
    return "\n\n".join(parts)


def ask(
    question: str,
    k: int = 8,
    scope: dict | None = None,
    *,
    domain_prompt: str | None = None,
) -> dict:
    """Answer a question using hybrid retrieval (FAISS + lexical) with graph context.

    Args:
        question: User question.
        k: Number of evidence chunks to retrieve.
        scope: Optional scope filter (collection/papers).
        domain_prompt: Custom system prompt domain line. Defaults to generic
            scientific assistant if not provided.
    """
    api_key = settings.effective_llm_api_key()
    base_url = settings.effective_llm_base_url()
    if not api_key:
        raise RuntimeError("LLM API key is required for /rag/ask (set DEEPSEEK_API_KEY or LLM_API_KEY)")

    allowed_sources = _allowed_paper_sources(scope)
    want = max(1, int(k))
    oversample = min(100, max(want, want * 5))

    # ── FAISS retrieval ──
    faiss_results: list[dict[str, Any]] = []
    try:
        store = load_faiss(str(latest_faiss_dir()))
        docs_and_scores = store.similarity_search_with_score(question, k=oversample)
        for doc, score in docs_and_scores:
            md = doc.metadata or {}
            snippet = (doc.page_content or "").strip()[:1200]
            if allowed_sources is not None:
                ps = str(md.get("paper_source") or "").strip()
                if not ps or ps not in allowed_sources:
                    continue
            faiss_results.append({
                "chunk_id": md.get("chunk_id"),
                "score": float(score),
                "paper_source": md.get("paper_source"),
                "md_path": md.get("md_path"),
                "start_line": md.get("start_line"),
                "end_line": md.get("end_line"),
                "section": md.get("section"),
                "kind": md.get("kind"),
                "snippet": snippet,
                "mode": "faiss",
            })
    except FileNotFoundError as e:
        raise FileNotFoundError("FAISS index not found. Please run full rebuild first.") from e
    except Exception as e:
        raise RuntimeError(f"FAISS retrieval failed: {e}") from e

    # ── Lexical retrieval (BM25-like) ──
    lexical_results: list[dict[str, Any]] = []
    try:
        run_dir = latest_run_dir(_runs_dir())
        chunks = load_chunks_from_run(run_dir)
        lex_hits = lexical_retrieve(question, chunks, k=oversample)
        for hit in lex_hits:
            if allowed_sources is not None:
                if not hit.paper_source or hit.paper_source not in allowed_sources:
                    continue
            lexical_results.append({
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "paper_source": hit.paper_source,
                "md_path": hit.md_path,
                "start_line": hit.start_line,
                "end_line": hit.end_line,
                "section": hit.section,
                "kind": hit.kind,
                "snippet": hit.snippet,
                "mode": "lexical",
            })
    except Exception:
        log.debug("Lexical retrieval unavailable, falling back to FAISS-only")

    # ── RRF fusion ──
    if lexical_results:
        fused = _rrf_fuse([faiss_results, lexical_results])
    else:
        fused = faiss_results

    # Build final evidence list
    evidence: list[dict[str, Any]] = []
    context_lines: list[str] = []
    for item in fused[:want]:
        paper_id = _paper_id_from_md_path(str(item.get("md_path") or "") or None)
        item["rank"] = len(evidence) + 1
        item["paper_id"] = paper_id
        evidence.append(item)
        context_lines.append(
            f"[E{len(evidence)}] {item.get('paper_source')} "
            f"{item.get('md_path')}:{item.get('start_line')}-{item.get('end_line')}\n"
            f"{item.get('snippet', '')}"
        )

    if allowed_sources is not None and len(evidence) < min(2, want):
        return {
            "answer": "",
            "evidence": evidence,
            "graph_context": None,
            "structured_knowledge": None,
            "insufficient_scope_evidence": True,
            "message": "当前范围内证据不足以回答该问题，请扩大范围或调整问题。",
        }

    # ── Graph context + structured knowledge ──
    graph_context = None
    structured_knowledge = None
    if evidence:
        paper_sources = list({e["paper_source"] for e in evidence if e.get("paper_source")})
        if paper_sources:
            try:
                with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
                    try:
                        graph_context = client.get_citation_context_by_paper_source(paper_sources, limit=50)
                    except Exception:
                        graph_context = None
                    try:
                        structured_knowledge = client.get_structured_knowledge_for_papers(paper_sources)
                    except Exception:
                        structured_knowledge = None
            except Exception:
                graph_context = None
                structured_knowledge = None

    # ── LLM generation (with graph context + structured knowledge in prompt) ──
    system = _build_system_prompt(domain_prompt)
    graph_block = _format_graph_context(graph_context)
    knowledge_block = _format_structured_knowledge(structured_knowledge)
    user_parts = [f"Question:\n{question}", "Evidence:\n" + "\n\n".join(context_lines)]
    if knowledge_block:
        user_parts.append(knowledge_block)
    if graph_block:
        user_parts.append(graph_block)
    user = "\n\n".join(user_parts)

    rag_timeout = max(10, min(180, int(settings.rag_llm_timeout_seconds)))
    rag_max_tokens = max(128, min(2048, int(settings.rag_llm_max_tokens)))
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=settings.llm_model,
        temperature=0,
        timeout=rag_timeout,
        max_tokens=rag_max_tokens,
        max_retries=0,
    )
    msg = llm.invoke([("system", system), ("user", user)])

    return {
        "answer": msg.content,
        "evidence": evidence,
        "graph_context": graph_context,
        "structured_knowledge": structured_knowledge,
        "retrieval_mode": "hybrid" if lexical_results else "faiss",
    }
