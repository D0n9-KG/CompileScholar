from __future__ import annotations

import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from langchain_core.embeddings import Embeddings

from app.graph.neo4j_client import Neo4jClient
from app.settings import settings


_TRANSIENT_HTTP_CODES = frozenset({429, 502, 503})
_TRANSIENT_KEYWORDS = frozenset(
    {
        'service unavailable',
        'rate limit',
        'overloaded',
        'too many requests',
        'bad gateway',
        'temporarily unavailable',
    }
)
_TRANSIENT_MAX = 8
_STABLE_MAX = 3
_STABLE_DELAY = 5.0
_FAISS_IMPORT_WARNING_PATTERNS = (
    r'.*SwigPyPacked.*',
    r'.*SwigPyObject.*',
    r'.*swigvarlink.*',
)


def _import_faiss() -> Any | None:
    try:
        with warnings.catch_warnings():
            for pattern in _FAISS_IMPORT_WARNING_PATTERNS:
                warnings.filterwarnings('ignore', message=pattern, category=DeprecationWarning)
            import faiss as faiss_module  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    return faiss_module


faiss = _import_faiss()


def _is_transient_error(exc: Exception) -> bool:
    for attr in ('status', 'status_code'):
        code = getattr(exc, attr, None)
        if code is not None:
            try:
                if int(code) in _TRANSIENT_HTTP_CODES:
                    return True
            except (TypeError, ValueError):
                pass

    response = getattr(exc, 'response', None)
    if response is not None:
        code = getattr(response, 'status_code', None)
        if code is not None:
            try:
                if int(code) in _TRANSIENT_HTTP_CODES:
                    return True
            except (TypeError, ValueError):
                pass

    matched = re.search(r'\b([45]\d{2})\b', str(exc))
    if matched and int(matched.group(1)) in _TRANSIENT_HTTP_CODES:
        return True

    msg = str(exc).lower()
    return any(keyword in msg for keyword in _TRANSIENT_KEYWORDS)


def _backoff_delay(attempt: int, base: float = 5.0, factor: float = 2.0, cap: float = 60.0) -> float:
    return min(base * math.pow(factor, attempt), cap)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _storage_similarity_root() -> Path:
    path = _backend_root() / settings.storage_dir / 'similarity'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _moves_dir() -> Path:
    path = _storage_similarity_root() / 'research_moves'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _items_path(kind: str) -> Path:
    if kind != 'move':
        raise ValueError(f'unknown kind: {kind}')
    return _moves_dir() / 'items.jsonl'


def _emb_path(kind: str) -> Path:
    if kind != 'move':
        raise ValueError(f'unknown kind: {kind}')
    return _moves_dir() / 'embeddings.npy'


def _meta_path(kind: str) -> Path:
    if kind != 'move':
        raise ValueError(f'unknown kind: {kind}')
    return _moves_dir() / 'meta.json'


def _neighbors_path(kind: str) -> Path:
    if kind != 'move':
        raise ValueError(f'unknown kind: {kind}')
    return _moves_dir() / 'neighbors.json'


def _embedding_client() -> Embeddings:
    from app.vector.faiss_store import _create_provider_compatible_embeddings

    return _create_provider_compatible_embeddings(max_retries=0)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


@dataclass(frozen=True)
class SimilarityItem:
    kind: str
    node_id: str
    paper_id: str
    text: str


def _read_items(kind: str) -> list[SimilarityItem]:
    path = _items_path(kind)
    if not path.exists():
        return []
    items: list[SimilarityItem] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            SimilarityItem(
                kind=str(row.get('kind') or kind),
                node_id=str(row.get('node_id') or ''),
                paper_id=str(row.get('paper_id') or ''),
                text=str(row.get('text') or ''),
            )
        )
    return items


def _write_items(kind: str, items: Iterable[SimilarityItem]) -> None:
    path = _items_path(kind)
    lines = [
        json.dumps(
            {
                'kind': item.kind,
                'node_id': item.node_id,
                'paper_id': item.paper_id,
                'text': item.text,
            },
            ensure_ascii=False,
            separators=(',', ':'),
        )
        for item in items
    ]
    tmp = path.with_suffix('.tmp')
    tmp.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    tmp.replace(path)


def _load_embeddings(kind: str) -> np.ndarray:
    path = _emb_path(kind)
    if not path.exists():
        raise FileNotFoundError(f'Missing similarity embeddings: {path}')
    data = np.load(str(path))
    if not isinstance(data, np.ndarray):
        raise RuntimeError('Invalid embeddings file')
    return data.astype(np.float32, copy=False)


def _save_embeddings(kind: str, x: np.ndarray) -> None:
    path = _emb_path(kind)
    tmp = path.with_suffix('.tmp.npy')
    np.save(str(tmp), x.astype(np.float32, copy=False))
    if not str(tmp).endswith('.npy'):
        tmp = Path(str(tmp) + '.npy')
    tmp.replace(path)


def _write_neighbors(kind: str, rows: list[dict[str, Any]]) -> None:
    path = _neighbors_path(kind)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def _build_index(x: np.ndarray) -> faiss.Index:
    if faiss is None:
        raise RuntimeError(
            'faiss library is not available; embedding-based similarity requires faiss. '
            'Install it with: pip install faiss-cpu'
        )
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError('Empty embedding matrix')
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
    if not source_indices:
        return []
    top_k = max(1, int(top_k))
    k = max(top_k + 1, min(len(items), max(top_k + 8, int(oversample))))
    scores, indices = index.search(x[source_indices], k)
    rows: list[dict[str, Any]] = []
    for row_index, src_index in enumerate(source_indices):
        src = items[src_index]
        targets: list[dict[str, Any]] = []
        for score, neighbor_index in zip(scores[row_index].tolist(), indices[row_index].tolist(), strict=False):
            if int(neighbor_index) < 0 or int(neighbor_index) == int(src_index):
                continue
            neighbor = items[int(neighbor_index)]
            if neighbor.paper_id == src.paper_id:
                continue
            if not src.node_id or not neighbor.node_id:
                continue
            targets.append({'target': neighbor.node_id, 'score': float(score)})
            if len(targets) >= top_k:
                break
        rows.append({'source': src.node_id, 'targets': targets})
    return rows


def _unique_tokens(values: object) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    if not isinstance(values, list):
        return ordered
    for value in values:
        token = str(value or '').strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _similarity_text_for_move(row: dict[str, Any]) -> str:
    summary = str(row.get('summary') or row.get('text') or '').strip()
    role = str(row.get('role') or '').strip()
    act_type = str(row.get('act_type') or '').strip()
    token_groups = [
        _unique_tokens(row.get('method_tokens')),
        _unique_tokens(row.get('object_tokens')),
        _unique_tokens(row.get('metric_tokens')),
        _unique_tokens(row.get('condition_tokens')),
        _unique_tokens(row.get('comparator_tokens')),
        _unique_tokens(row.get('effect_directions')),
        _unique_tokens(row.get('limitation_tokens')),
        _unique_tokens(row.get('resource_tokens')),
    ]
    token_lines = [' '.join(group) for group in token_groups if group]
    parts = [summary]
    if role:
        parts.append(f'role: {role}')
    if act_type:
        parts.append(f'act: {act_type}')
    parts.extend(token_lines)
    return '\n'.join(part for part in parts if part).strip()


def _move_similarity_items(rows: Iterable[dict[str, Any]]) -> list[SimilarityItem]:
    items: list[SimilarityItem] = []
    for row in rows:
        move_id = str(row.get('move_id') or row.get('source_id') or row.get('id') or '').strip()
        paper_id = str(row.get('paper_id') or '').strip()
        text = _similarity_text_for_move(dict(row))
        if not move_id or not paper_id or not text:
            continue
        items.append(
            SimilarityItem(
                kind='research_move',
                node_id=move_id,
                paper_id=paper_id,
                text=text,
            )
        )
    return items


def _embed_items(embed: Embeddings, items: list[SimilarityItem]) -> np.ndarray:
    texts = [item.text for item in items]
    vectors = embed.embed_documents(texts)
    return _normalize_rows(np.array(vectors, dtype=np.float32))


def _persist_move_store(
    *,
    items: list[SimilarityItem],
    x: np.ndarray,
    model: str,
    built_at: str,
    top_k: int,
) -> tuple[int, int]:
    mode = 'embedding'
    _write_items('move', items)
    _save_embeddings('move', x)
    _meta_path('move').write_text(
        json.dumps(
            {
                'built_at': built_at,
                'model': model,
                'mode': mode,
            },
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    if not items:
        _write_neighbors('move', [])
        return 0, 0
    index = _build_index(x)
    neighbors = _topk_pairs(index, x, items, list(range(len(items))), top_k=top_k)
    _write_neighbors('move', neighbors)
    edge_count = sum(len(row.get('targets') or []) for row in neighbors)
    return len(neighbors), edge_count


def rebuild_similarity_global(
    progress: callable | None = None,  # noqa: ANN001
    log: callable | None = None,  # noqa: ANN001
    logic_top_k: int = 20,
) -> dict[str, Any]:
    progress = progress or (lambda stage, p, msg=None: None)
    log = log or (lambda line: None)

    model = str(settings.effective_embedding_model() or '')
    built_at = _utc_now_iso()

    progress('similarity:fetch', 0.08, 'Fetching ResearchMove texts from PaperLogicTrace rows')
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        moves = _move_similarity_items(client.list_research_moves(limit=50000) or [])

    if faiss is None:
        raise RuntimeError(
            'faiss library is not available; embedding-based similarity requires faiss. '
            'Install it with: pip install faiss-cpu'
        )

    embed = _embedding_client()
    attempt = 0
    while True:
        try:
            progress('similarity:embed_moves', 0.32, f'Embedding {len(moves)} research moves (attempt {attempt + 1})')
            move_x = _embed_items(embed, moves) if moves else np.zeros((0, 0), dtype=np.float32)
            break
        except Exception as exc:  # noqa: BLE001
            transient = _is_transient_error(exc)
            max_tries = _TRANSIENT_MAX if transient else _STABLE_MAX
            error_label = 'transient' if transient else 'stable'
            attempt += 1
            if attempt < max_tries:
                wait = _backoff_delay(attempt - 1) if transient else _STABLE_DELAY
                log(
                    f'Embedding attempt {attempt}/{max_tries} failed [{error_label}]: '
                    f'{str(exc).strip()}. Retrying in {wait:.0f}s...'
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f'Embedding failed after {attempt} attempts [{error_label}]: {str(exc).strip()}'
            ) from exc

    progress('similarity:neighbors', 0.76, 'Computing ResearchMove neighbors')
    sources_written, edges_written = _persist_move_store(
        items=moves,
        x=move_x,
        model=model,
        built_at=built_at,
        top_k=logic_top_k,
    )
    progress('similarity:done', 1.0, 'ResearchMove similarity rebuild done')
    log(f'similarity rebuilt: research_moves={len(moves)} model={model}')
    return {
        'ok': True,
        'built_at': built_at,
        'model': model,
        'mode': 'embedding',
        'research_moves': len(moves),
        'neighbor_sources': int(sources_written),
        'neighbor_edges': int(edges_written),
    }


def update_similarity_for_paper(
    paper_id: str,
    progress: callable | None = None,  # noqa: ANN001
    log: callable | None = None,  # noqa: ANN001
    logic_top_k: int = 20,
) -> dict[str, Any]:
    progress = progress or (lambda stage, p, msg=None: None)
    log = log or (lambda line: None)
    pid = str(paper_id or '').strip()
    if not pid:
        raise ValueError('paper_id required')

    progress('similarity:update:load', 0.05, 'Loading ResearchMove similarity store')
    if not _items_path('move').exists() or not _meta_path('move').exists() or not _emb_path('move').exists():
        return rebuild_similarity_global(progress=progress, log=log, logic_top_k=logic_top_k)

    try:
        meta = json.loads(_meta_path('move').read_text(encoding='utf-8') or '{}')
    except Exception:
        meta = {}
    if str(meta.get('mode') or '') != 'embedding':
        return rebuild_similarity_global(progress=progress, log=log, logic_top_k=logic_top_k)
    if faiss is None:
        return rebuild_similarity_global(progress=progress, log=log, logic_top_k=logic_top_k)

    embed = _embedding_client()
    items = _read_items('move')
    x = _load_embeddings('move')
    idx_map = {item.node_id: index for index, item in enumerate(items)}

    progress('similarity:update:fetch', 0.16, f'Fetching ResearchMove texts for {pid}')
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        updates = _move_similarity_items(client.list_research_moves(paper_id=pid, limit=50000) or [])

    changed: list[int] = []
    if updates:
        to_embed = [item for item in updates if item.text.strip()]
        vectors: list[list[float]] = []
        attempt = 0
        while to_embed:
            try:
                progress('similarity:update:embed', 0.38, f'Embedding {len(to_embed)} updated research moves')
                vectors = embed.embed_documents([item.text for item in to_embed])
                break
            except Exception as exc:  # noqa: BLE001
                transient = _is_transient_error(exc)
                max_tries = _TRANSIENT_MAX if transient else _STABLE_MAX
                error_label = 'transient' if transient else 'stable'
                attempt += 1
                if attempt < max_tries:
                    wait = _backoff_delay(attempt - 1) if transient else _STABLE_DELAY
                    log(
                        f'Similarity update embedding attempt {attempt}/{max_tries} failed ['
                        f'{error_label}]: {str(exc).strip()}. Retrying in {wait:.0f}s...'
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f'Similarity update failed: embedding unavailable after {attempt} attempts '
                    f'[{error_label}]. Error: {str(exc).strip()}'
                ) from exc

        vector_map: dict[str, np.ndarray] = {}
        if vectors:
            normalized = _normalize_rows(np.array(vectors, dtype=np.float32))
            vector_map = {item.node_id: normalized[index] for index, item in enumerate(to_embed)}

        dim = int(x.shape[1]) if x.ndim == 2 and x.shape[1] else (int(next(iter(vector_map.values())).shape[0]) if vector_map else 0)
        if dim <= 0:
            raise RuntimeError('Failed to infer embedding dimension for similarity update')
        if x.ndim != 2 or (x.shape[0] and x.shape[1] != dim):
            raise RuntimeError('Similarity store dimension mismatch; rebuild required')
        if x.ndim != 2:
            x = np.zeros((0, dim), dtype=np.float32)

        pending_rows: list[np.ndarray] = []
        base_rows = int(x.shape[0])
        for update in updates:
            node_id = update.node_id
            if not node_id:
                continue
            vector = vector_map.get(node_id)
            if node_id in idx_map:
                index = idx_map[node_id]
                items[index] = update
                if index < base_rows:
                    x[index] = vector if vector is not None else np.zeros((dim,), dtype=np.float32)
                else:
                    pending_rows[index - base_rows] = (
                        vector.reshape(1, -1) if vector is not None else np.zeros((1, dim), dtype=np.float32)
                    )
                changed.append(index)
                continue
            idx_map[node_id] = len(items)
            items.append(update)
            pending_rows.append(vector.reshape(1, -1) if vector is not None else np.zeros((1, dim), dtype=np.float32))
            changed.append(len(items) - 1)

        if pending_rows:
            x = np.vstack([x, np.vstack(pending_rows)])

    progress('similarity:update:neighbors', 0.72, 'Refreshing ResearchMove neighbor cache')
    sources_written, edges_written = _persist_move_store(
        items=items,
        x=x,
        model=str(settings.effective_embedding_model() or ''),
        built_at=_utc_now_iso(),
        top_k=logic_top_k,
    )
    progress('similarity:update:done', 1.0, 'ResearchMove similarity update done')
    log(f'similarity updated for {pid}: research_moves={len(set(changed))}')
    return {
        'ok': True,
        'paper_id': pid,
        'built_at': _utc_now_iso(),
        'model': str(settings.effective_embedding_model() or ''),
        'mode': 'embedding',
        'research_moves_updated': len(set(changed)),
        'neighbor_sources': int(sources_written),
        'neighbor_edges': int(edges_written),
    }
