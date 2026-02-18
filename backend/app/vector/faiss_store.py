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

    # Incremental FAISS build: batch-level retry to avoid restarting all chunks on transient errors.
    batch_size = 64
    max_batch_retries = 5
    base_retry_delay = 5  # seconds (exponential backoff: 5, 10, 20, 40, 80)
    total_batches = (len(texts) + batch_size - 1) // batch_size
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

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]

        for attempt in range(max_batch_retries):
            try:
                # First batch initializes the FAISS store; subsequent batches append incrementally.
                if store is None:
                    store = FAISS.from_texts(texts=batch_texts, embedding=embeddings, metadatas=batch_metadatas)
                else:
                    store.add_texts(texts=batch_texts, metadatas=batch_metadatas)

                print(f"批次 {batch_idx + 1}/{total_batches} 已完成")
                break
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc).strip()
                retryable = _is_retryable_embedding_error(exc)
                retry_delay = base_retry_delay * (2 ** attempt)
                if retryable and attempt < max_batch_retries - 1:
                    print(
                        f"FAISS batch {batch_idx + 1}/{total_batches} "
                        f"attempt {attempt + 1}/{max_batch_retries} failed: {error_msg}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    reason = (
                        f"non-retryable embedding error at batch {batch_idx + 1}/{total_batches}, "
                        f"attempt {attempt + 1}/{max_batch_retries}"
                        if not retryable
                        else f"embedding unavailable at batch {batch_idx + 1}/{total_batches} "
                        f"after {max_batch_retries} attempts"
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
