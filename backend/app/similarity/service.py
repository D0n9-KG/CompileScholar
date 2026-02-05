from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import faiss  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore[assignment]
import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.graph.neo4j_client import Neo4jClient
from app.settings import settings


_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]+")


def _tokens(s: str) -> list[str]:
    toks = _TOKEN_RE.findall((s or "").lower())
    out: list[str] = []
    for t in toks:
        if re.fullmatch(r"[\u4e00-\u9fff]+", t) and len(t) > 2:
            out.extend(list(t))
        else:
            out.append(t)
    return [t for t in out if t and t not in {"the", "and", "of", "to", "in", "a", "an"}]


def _lexical_topk_batch(items: list["SimilarityItem"], top_k: int) -> list[dict[str, Any]]:
    """
    Lexical fallback when embeddings are unavailable.
    Uses a token inverted index + cosine over binary token vectors.
    """
    top_k = max(1, int(top_k))
    toks = [set(_tokens(it.text)) for it in items]
    inv: dict[str, list[int]] = {}
    for i, ts in enumerate(toks):
        for t in ts:
            inv.setdefault(t, []).append(i)

    out: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        ts = toks[i]
        if not ts:
            out.append({"source": it.node_id, "targets": []})
            continue
        cand: dict[int, int] = {}
        for t in ts:
            for j in inv.get(t, []):
                if j == i:
                    continue
                cand[j] = cand.get(j, 0) + 1
        scored: list[tuple[float, int]] = []
        denom_i = float(max(1, len(ts)))
        for j, inter in cand.items():
            if items[j].paper_id == it.paper_id:
                continue
            denom = (denom_i * float(max(1, len(toks[j])))) ** 0.5
            score = float(inter) / denom if denom else 0.0
            if score <= 0:
                continue
            scored.append((score, j))
        scored.sort(key=lambda x: x[0], reverse=True)
        targets = [{"target": items[j].node_id, "score": float(score)} for score, j in scored[:top_k]]
        out.append({"source": it.node_id, "targets": targets})
    return out


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _backend_root() -> Path:
    # backend/app/similarity/service.py -> backend/
    return Path(__file__).resolve().parents[2]


def _storage_similarity_root() -> Path:
    p = _backend_root() / settings.storage_dir / "similarity"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _claims_dir() -> Path:
    p = _storage_similarity_root() / "claims"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _logic_dir() -> Path:
    p = _storage_similarity_root() / "logic_steps"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _items_path(kind: str) -> Path:
    if kind == "claim":
        return _claims_dir() / "items.jsonl"
    if kind == "logic":
        return _logic_dir() / "items.jsonl"
    raise ValueError(f"unknown kind: {kind}")


def _emb_path(kind: str) -> Path:
    if kind == "claim":
        return _claims_dir() / "embeddings.npy"
    if kind == "logic":
        return _logic_dir() / "embeddings.npy"
    raise ValueError(f"unknown kind: {kind}")


def _meta_path(kind: str) -> Path:
    if kind == "claim":
        return _claims_dir() / "meta.json"
    if kind == "logic":
        return _logic_dir() / "meta.json"
    raise ValueError(f"unknown kind: {kind}")


def _embedding_client() -> OpenAIEmbeddings:
    api_key = settings.effective_embedding_api_key()
    base_url = settings.effective_embedding_base_url()
    model = settings.effective_embedding_model()
    if not model:
        raise RuntimeError("EMBEDDING_MODEL is not set; similarity disabled")
    if not api_key:
        raise RuntimeError("Embedding API key is required to build similarity indexes")
    return OpenAIEmbeddings(api_key=api_key, base_url=base_url, model=model, chunk_size=64)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


@dataclass(frozen=True)
class SimilarityItem:
    kind: str  # "claim" | "logic"
    node_id: str  # claim_id or logic_step_id
    paper_id: str
    text: str


def _read_items(kind: str) -> list[SimilarityItem]:
    p = _items_path(kind)
    if not p.exists():
        return []
    out: list[SimilarityItem] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            SimilarityItem(
                kind=str(d.get("kind") or kind),
                node_id=str(d["node_id"]),
                paper_id=str(d["paper_id"]),
                text=str(d.get("text") or ""),
            )
        )
    return out


def _write_items(kind: str, items: Iterable[SimilarityItem]) -> None:
    p = _items_path(kind)
    lines = []
    for it in items:
        lines.append(
            json.dumps(
                {"kind": it.kind, "node_id": it.node_id, "paper_id": it.paper_id, "text": it.text},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    tmp = p.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(p)


def _load_embeddings(kind: str) -> np.ndarray:
    p = _emb_path(kind)
    if not p.exists():
        raise FileNotFoundError(f"Missing similarity embeddings: {p}")
    x = np.load(str(p))
    if not isinstance(x, np.ndarray):
        raise RuntimeError("Invalid embeddings file")
    return x.astype(np.float32, copy=False)


def _save_embeddings(kind: str, x: np.ndarray) -> None:
    p = _emb_path(kind)
    tmp = p.with_suffix(".tmp.npy")
    np.save(str(tmp), x.astype(np.float32, copy=False))
    # np.save appends .npy if missing; ensure consistent final name
    if not str(tmp).endswith(".npy"):
        tmp = Path(str(tmp) + ".npy")
    tmp.replace(p)


def _build_index(x: np.ndarray) -> faiss.Index:
    if faiss is None:
        raise RuntimeError("faiss is not available")
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Empty embedding matrix")
    dim = int(x.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(x)
    return index


def _topk_pairs(
    index: faiss.Index,
    x: np.ndarray,
    items: list[SimilarityItem],
    source_indices: list[int],
    top_k: int,
    oversample: int = 64,
) -> list[dict[str, Any]]:
    """
    Return a list of batch items suitable for Neo4j upsert:
      { "source": node_id, "targets": [ {"target": node_id, "score": float}, ... ] }
    Only cross-paper neighbors are kept.
    """
    if not source_indices:
        return []
    top_k = max(1, int(top_k))
    k = max(top_k + 1, min(len(items), max(top_k + 8, int(oversample))))

    sources_x = x[source_indices]
    D, I = index.search(sources_x, k)
    out: list[dict[str, Any]] = []
    for row_idx, src_i in enumerate(source_indices):
        src = items[src_i]
        targets: list[dict[str, Any]] = []
        for score, nbr_i in zip(D[row_idx].tolist(), I[row_idx].tolist(), strict=False):
            if int(nbr_i) < 0:
                continue
            if int(nbr_i) == int(src_i):
                continue
            nbr = items[int(nbr_i)]
            if nbr.paper_id == src.paper_id:
                continue
            if not nbr.node_id or not src.node_id:
                continue
            targets.append({"target": nbr.node_id, "score": float(score)})
            if len(targets) >= top_k:
                break
        out.append({"source": src.node_id, "targets": targets})
    return out


def rebuild_similarity_global(
    progress: callable | None = None,  # noqa: ANN001
    log: callable | None = None,  # noqa: ANN001
    claim_top_k: int = 30,
    logic_top_k: int = 20,
) -> dict[str, Any]:
    """
    Full rebuild:
    - fetch effective texts for all Claim/LogicStep nodes
    - embed all texts and persist embeddings
    - compute Top-K cross-paper neighbors and write SIMILAR_* edges to Neo4j
    """
    progress = progress or (lambda stage, p, msg=None: None)
    log = log or (lambda line: None)

    model = str(settings.effective_embedding_model() or "")
    built_at = _utc_now_iso()

    progress("similarity:fetch", 0.05, "Fetching effective texts from Neo4j")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        claims = [
            SimilarityItem(kind="claim", node_id=str(r["node_id"]), paper_id=str(r["paper_id"]), text=str(r["text"]))
            for r in (client.list_claim_similarity_rows() or [])
        ]
        logic = [
            SimilarityItem(kind="logic", node_id=str(r["node_id"]), paper_id=str(r["paper_id"]), text=str(r["text"]))
            for r in (client.list_logic_step_similarity_rows() or [])
        ]

    def _embed_items(kind: str, items: list[SimilarityItem]) -> np.ndarray:
        texts = [it.text for it in items]
        # embed_documents returns List[List[float]]
        vecs = embed.embed_documents(texts)
        x = np.array(vecs, dtype=np.float32)
        return _normalize_rows(x)

    mode = "lexical"
    claim_x = np.zeros((0, 0), dtype=np.float32)
    logic_x = np.zeros((0, 0), dtype=np.float32)
    try:
        if faiss is None:
            raise RuntimeError("faiss is not available")
        embed = _embedding_client()
        progress("similarity:embed_claims", 0.15, f"Embedding claims ({len(claims)})")
        claim_x = _embed_items("claim", claims) if claims else np.zeros((0, 0), dtype=np.float32)
        progress("similarity:embed_logic", 0.30, f"Embedding logic steps ({len(logic)})")
        logic_x = _embed_items("logic", logic) if logic else np.zeros((0, 0), dtype=np.float32)
        mode = "embedding"
    except Exception as exc:  # noqa: BLE001
        # Common in local/dev environments: embedding provider quota/billing issues, missing keys, etc.
        mode = "lexical"
        claim_x = np.zeros((0, 0), dtype=np.float32)
        logic_x = np.zeros((0, 0), dtype=np.float32)
        log(f"similarity embedding mode unavailable; falling back to lexical mode: {exc}")

    if claims:
        _write_items("claim", claims)
        if mode == "embedding":
            _save_embeddings("claim", claim_x)
        _meta_path("claim").write_text(
            json.dumps({"built_at": built_at, "model": model, "mode": mode}, ensure_ascii=False),
            encoding="utf-8",
        )
    if logic:
        _write_items("logic", logic)
        if mode == "embedding":
            _save_embeddings("logic", logic_x)
        _meta_path("logic").write_text(
            json.dumps({"built_at": built_at, "model": model, "mode": mode}, ensure_ascii=False),
            encoding="utf-8",
        )

    # Build indexes and compute neighbors
    progress("similarity:neighbors_claims", 0.55, "Computing claim neighbors")
    if claims:
        if mode == "embedding":
            claim_index = _build_index(claim_x)
            batch = _topk_pairs(claim_index, claim_x, claims, list(range(len(claims))), top_k=claim_top_k)
        else:
            batch = _lexical_topk_batch(claims, top_k=claim_top_k)
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            client.replace_similar_claim_edges_batch(batch, model=model, built_at=built_at)

    progress("similarity:neighbors_logic", 0.75, "Computing logic-step neighbors")
    if logic:
        if mode == "embedding":
            logic_index = _build_index(logic_x)
            batch = _topk_pairs(logic_index, logic_x, logic, list(range(len(logic))), top_k=logic_top_k)
        else:
            batch = _lexical_topk_batch(logic, top_k=logic_top_k)
        with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
            client.replace_similar_logic_edges_batch(batch, model=model, built_at=built_at)

    progress("similarity:done", 1.0, "Similarity rebuild done")
    log(f"similarity rebuilt: mode={mode} claims={len(claims)} logic_steps={len(logic)} model={model}")
    return {"ok": True, "built_at": built_at, "model": model, "mode": mode, "claims": len(claims), "logic_steps": len(logic)}


def update_similarity_for_paper(
    paper_id: str,
    progress: callable | None = None,  # noqa: ANN001
    log: callable | None = None,  # noqa: ANN001
    claim_top_k: int = 30,
    logic_top_k: int = 20,
) -> dict[str, Any]:
    """
    Incremental update for one paper:
    - load stored embeddings/items
    - fetch current effective texts for that paper
    - re-embed those nodes and update their rows
    - recompute Top-K neighbors for those nodes only, and upsert edges

    Note: requires a previous global rebuild to create the on-disk embeddings store.
    """
    progress = progress or (lambda stage, p, msg=None: None)
    log = log or (lambda line: None)
    pid = str(paper_id or "").strip()
    if not pid:
        raise ValueError("paper_id required")

    model = str(settings.effective_embedding_model() or "")
    built_at = _utc_now_iso()

    progress("similarity:update:load", 0.05, "Loading similarity stores")
    if not _items_path("claim").exists() or not _items_path("logic").exists() or not _meta_path("claim").exists() or not _meta_path("logic").exists():
        return rebuild_similarity_global(progress=progress, log=log, claim_top_k=claim_top_k, logic_top_k=logic_top_k)
    try:
        claim_meta = json.loads(_meta_path("claim").read_text(encoding="utf-8") or "{}")
        logic_meta = json.loads(_meta_path("logic").read_text(encoding="utf-8") or "{}")
    except Exception:
        claim_meta = {}
        logic_meta = {}
    if str(claim_meta.get("mode") or "") != "embedding" or str(logic_meta.get("mode") or "") != "embedding":
        return rebuild_similarity_global(progress=progress, log=log, claim_top_k=claim_top_k, logic_top_k=logic_top_k)
    if not _emb_path("claim").exists() or not _emb_path("logic").exists():
        return rebuild_similarity_global(progress=progress, log=log, claim_top_k=claim_top_k, logic_top_k=logic_top_k)
    if faiss is None:
        return rebuild_similarity_global(progress=progress, log=log, claim_top_k=claim_top_k, logic_top_k=logic_top_k)
    try:
        embed = _embedding_client()
    except Exception as exc:  # noqa: BLE001
        log(f"similarity update embedding unavailable; falling back to rebuild: {exc}")
        return rebuild_similarity_global(progress=progress, log=log, claim_top_k=claim_top_k, logic_top_k=logic_top_k)
    claim_items = _read_items("claim")
    logic_items = _read_items("logic")
    claim_x = _load_embeddings("claim")
    logic_x = _load_embeddings("logic")

    claim_idx = {it.node_id: i for i, it in enumerate(claim_items)}
    logic_idx = {it.node_id: i for i, it in enumerate(logic_items)}

    progress("similarity:update:fetch", 0.12, f"Fetching effective texts for {pid}")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        new_claims = [
            SimilarityItem(kind="claim", node_id=str(r["node_id"]), paper_id=str(r["paper_id"]), text=str(r["text"]))
            for r in (client.list_claim_similarity_rows(paper_id=pid) or [])
        ]
        new_logic = [
            SimilarityItem(kind="logic", node_id=str(r["node_id"]), paper_id=str(r["paper_id"]), text=str(r["text"]))
            for r in (client.list_logic_step_similarity_rows(paper_id=pid) or [])
        ]

    def _apply_updates(
        kind: str, items: list[SimilarityItem], x: np.ndarray, idx_map: dict[str, int], updates: list[SimilarityItem]
    ):
        changed: list[int] = []
        if not updates:
            return items, x, changed

        # Embed only non-empty texts to avoid provider errors.
        to_embed: list[SimilarityItem] = [u for u in updates if (u.text or "").strip()]
        vecs = embed.embed_documents([u.text for u in to_embed]) if to_embed else []
        u_x = _normalize_rows(np.array(vecs, dtype=np.float32)) if to_embed else np.zeros((0, x.shape[1]), dtype=np.float32)

        dim = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else (int(u_x.shape[1]) if u_x.ndim == 2 and u_x.shape[1] else 0)
        if dim <= 0:
            raise RuntimeError("Failed to infer embedding dimension for similarity update")
        if x.ndim != 2 or x.shape[1] != dim:
            raise RuntimeError("Similarity store dimension mismatch; rebuild required")

        vec_by_id = {u.node_id: u_x[i] for i, u in enumerate(to_embed)}

        for u in updates:
            node_id = u.node_id
            if not node_id:
                continue
            vec = vec_by_id.get(node_id)
            if node_id in idx_map:
                i = idx_map[node_id]
                items[i] = SimilarityItem(kind=kind, node_id=node_id, paper_id=u.paper_id, text=u.text)
                if vec is not None:
                    x[i] = vec
                else:
                    x[i] = np.zeros((dim,), dtype=np.float32)
                changed.append(i)
            else:
                idx_map[node_id] = len(items)
                items.append(SimilarityItem(kind=kind, node_id=node_id, paper_id=u.paper_id, text=u.text))
                row = vec.reshape(1, -1) if vec is not None else np.zeros((1, dim), dtype=np.float32)
                x = np.vstack([x, row])
                changed.append(len(items) - 1)
        return items, x, changed

    progress("similarity:update:embed_claims", 0.25, "Embedding updated claims")
    claim_items, claim_x, claim_changed = _apply_updates("claim", claim_items, claim_x, claim_idx, new_claims)
    progress("similarity:update:embed_logic", 0.35, "Embedding updated logic steps")
    logic_items, logic_x, logic_changed = _apply_updates("logic", logic_items, logic_x, logic_idx, new_logic)

    progress("similarity:update:neighbors", 0.55, "Computing updated neighbors")
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        if claim_items and claim_changed:
            changed_idx = sorted(set(claim_changed))
            active = [i for i in changed_idx if (claim_items[i].text or "").strip()]
            cleared = [i for i in changed_idx if not (claim_items[i].text or "").strip()]
            batch = []
            if active:
                idx = _build_index(claim_x)
                batch.extend(_topk_pairs(idx, claim_x, claim_items, active, top_k=claim_top_k))
            for i in cleared:
                batch.append({"source": claim_items[i].node_id, "targets": []})
            client.replace_similar_claim_edges_batch(batch, model=model, built_at=built_at)
        if logic_items and logic_changed:
            changed_idx = sorted(set(logic_changed))
            active = [i for i in changed_idx if (logic_items[i].text or "").strip()]
            cleared = [i for i in changed_idx if not (logic_items[i].text or "").strip()]
            batch = []
            if active:
                idx = _build_index(logic_x)
                batch.extend(_topk_pairs(idx, logic_x, logic_items, active, top_k=logic_top_k))
            for i in cleared:
                batch.append({"source": logic_items[i].node_id, "targets": []})
            client.replace_similar_logic_edges_batch(batch, model=model, built_at=built_at)

    progress("similarity:update:save", 0.85, "Saving similarity stores")
    if claim_items:
        _write_items("claim", claim_items)
        _save_embeddings("claim", claim_x)
        _meta_path("claim").write_text(
            json.dumps({"built_at": built_at, "model": model, "mode": "embedding"}, ensure_ascii=False),
            encoding="utf-8",
        )
    if logic_items:
        _write_items("logic", logic_items)
        _save_embeddings("logic", logic_x)
        _meta_path("logic").write_text(
            json.dumps({"built_at": built_at, "model": model, "mode": "embedding"}, ensure_ascii=False),
            encoding="utf-8",
        )

    progress("similarity:update:done", 1.0, "Similarity update done")
    log(f"similarity updated for {pid}: claims={len(claim_changed)} logic={len(logic_changed)}")
    return {
        "ok": True,
        "paper_id": pid,
        "built_at": built_at,
        "model": model,
        "claims_updated": len(set(claim_changed)),
        "logic_steps_updated": len(set(logic_changed)),
    }
