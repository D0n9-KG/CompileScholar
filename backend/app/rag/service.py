from __future__ import annotations

from pathlib import Path

from langchain_openai import ChatOpenAI

from app.graph.neo4j_client import Neo4jClient
from app.ingest.paper_meta import load_canonical_meta
from app.settings import settings
from app.vector.faiss_store import load_faiss
from app.rag.retrieval import latest_run_dir, load_chunks_from_run, lexical_retrieve


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


def ask(question: str, k: int = 8, scope: dict | None = None) -> dict:
    api_key = settings.effective_llm_api_key()
    base_url = settings.effective_llm_base_url()
    if not api_key:
        raise RuntimeError("LLM API key is required for /rag/ask (set DEEPSEEK_API_KEY or LLM_API_KEY)")

    evidence = []
    context_lines = []
    allowed_sources = _allowed_paper_sources(scope)
    want = max(1, int(k))
    # Oversample before scope filtering to reduce "scope starvation".
    oversample = min(100, max(want, want * 5))

    # Prefer FAISS if available; fall back to lexical retrieval (still uses LLM for synthesis).
    try:
        store = load_faiss(str(latest_faiss_dir()))
        docs_and_scores = store.similarity_search_with_score(question, k=oversample)
        for idx, (doc, score) in enumerate(docs_and_scores, start=1):
            md = doc.metadata or {}
            snippet = (doc.page_content or "").strip()
            snippet = snippet[:1200]
            if allowed_sources is not None:
                ps = str(md.get("paper_source") or "").strip()
                if not ps or ps not in allowed_sources:
                    continue
            paper_id = _paper_id_from_md_path(str(md.get("md_path") or "") or None)
            evidence.append(
                {
                    "rank": len(evidence) + 1,
                    "score": float(score),
                    "chunk_id": md.get("chunk_id"),
                    "paper_id": paper_id,
                    "paper_source": md.get("paper_source"),
                    "md_path": md.get("md_path"),
                    "start_line": md.get("start_line"),
                    "end_line": md.get("end_line"),
                    "section": md.get("section"),
                    "kind": md.get("kind"),
                    "snippet": snippet,
                    "mode": "faiss",
                }
            )
            context_lines.append(
                f"[E{len(evidence)}] {md.get('paper_source')} {md.get('md_path')}:{md.get('start_line')}-{md.get('end_line')}\n{snippet}"
            )
            if len(evidence) >= want:
                break
    except Exception:
        run_dir = latest_run_dir(_runs_dir())
        chunks = load_chunks_from_run(run_dir)
        retrieved = lexical_retrieve(question, chunks, k=oversample)
        for idx, r in enumerate(retrieved, start=1):
            if allowed_sources is not None:
                ps = str(r.paper_source or "").strip()
                if not ps or ps not in allowed_sources:
                    continue
            paper_id = _paper_id_from_md_path(r.md_path)
            evidence.append(
                {
                    "rank": len(evidence) + 1,
                    "score": r.score,
                    "chunk_id": r.chunk_id,
                    "paper_id": paper_id,
                    "paper_source": r.paper_source,
                    "md_path": r.md_path,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "section": r.section,
                    "kind": r.kind,
                    "snippet": r.snippet,
                    "mode": "lexical",
                }
            )
            context_lines.append(
                f"[E{len(evidence)}] {r.paper_source} {r.md_path}:{r.start_line}-{r.end_line}\n{r.snippet}"
            )
            if len(evidence) >= want:
                break

    if allowed_sources is not None and len(evidence) < min(2, want):
        return {
            "answer": "",
            "evidence": evidence,
            "graph_context": None,
            "insufficient_scope_evidence": True,
            "message": "当前范围内证据不足以回答该问题，请扩大范围或调整问题。",
        }

    graph_context = None
    if evidence:
        try:
            paper_sources = list({e["paper_source"] for e in evidence if e.get("paper_source")})
            with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
                graph_context = client.get_citation_context_by_paper_source(paper_sources, limit=50)
        except Exception:
            graph_context = None

    system = (
        "You are a mechanics research assistant. Answer ONLY using the provided evidence snippets.\n"
        "If evidence is insufficient, say what is missing.\n"
        "Cite evidence by referencing the evidence ids like [E1], [E2]."
    )
    user = f"Question:\n{question}\n\nEvidence:\n" + "\n\n".join(context_lines)

    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=settings.llm_model, temperature=0)
    msg = llm.invoke([("system", system), ("user", user)])

    return {"answer": msg.content, "evidence": evidence, "graph_context": graph_context}
