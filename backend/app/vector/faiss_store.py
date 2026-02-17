from __future__ import annotations

import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from app.ingest.models import Chunk
from app.settings import settings


def build_faiss_for_chunks(chunks: list[Chunk], out_dir: str) -> dict:
    api_key = settings.effective_embedding_api_key()
    base_url = settings.effective_embedding_base_url()
    model = settings.effective_embedding_model()
    if not model:
        raise RuntimeError("EMBEDDING_MODEL is not set; FAISS disabled")
    if not api_key:
        raise RuntimeError(
            "Embedding API key is required to build FAISS index (set EMBEDDING_PROVIDER=siliconflow and SILICONFLOW_API_KEY)"
        )
    if not chunks:
        raise RuntimeError("FAISS index build failed: no chunks available to index")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Some providers (e.g., SiliconFlow) limit embedding batch size (e.g., 64).
    # LangChain's OpenAIEmbeddings supports chunk_size to control embed_documents batching.
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        chunk_size=64,
    )

    texts = [c.text for c in chunks]
    metadatas = [
        {
            "chunk_id": c.chunk_id,
            "paper_source": c.paper_source,
            "md_path": c.md_path,
            "start_line": c.span.start_line,
            "end_line": c.span.end_line,
            "section": c.section,
            "kind": c.kind,
        }
        for c in chunks
    ]

    # Retry logic for embedding API (handles transient 502 errors)
    max_retries = 3
    retry_delay = 5  # seconds
    store = None

    def _is_retryable_embedding_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code in {408, 429, 500, 502, 503, 504}
        error_text = str(exc).lower()
        transient_signals = (
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "rate limit",
        )
        return any(signal in error_text for signal in transient_signals)

    for attempt in range(max_retries):
        try:
            store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
            break  # Success - exit retry loop
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc).strip()
            retryable = _is_retryable_embedding_error(exc)
            if retryable and attempt < max_retries - 1:
                print(f"FAISS build attempt {attempt + 1}/{max_retries} failed: {error_msg}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                reason = (
                    f"non-retryable embedding error on attempt {attempt + 1}/{max_retries}"
                    if not retryable
                    else f"embedding unavailable after {max_retries} attempts"
                )
                raise RuntimeError(
                    f"FAISS index build failed: {reason}. "
                    f"Error: {error_msg}. Please check embedding API configuration and try again."
                ) from exc

    if store is None:
        raise RuntimeError("FAISS index build failed: retry loop exited without creating an index")
    store.save_local(str(out))
    return {"chunks_indexed": len(chunks), "dir": str(out)}


def load_faiss(out_dir: str) -> FAISS:
    api_key = settings.effective_embedding_api_key()
    base_url = settings.effective_embedding_base_url()
    model = settings.effective_embedding_model()
    if not model:
        raise RuntimeError("EMBEDDING_MODEL is not set; FAISS disabled")
    if not api_key:
        raise RuntimeError("Embedding API key is required to load FAISS index")
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        chunk_size=64,
    )
    return FAISS.load_local(out_dir, embeddings, allow_dangerous_deserialization=True)
