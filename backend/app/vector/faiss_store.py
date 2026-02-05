from __future__ import annotations

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

    store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
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
