"""Local textbook graph extractor.

This module builds a lightweight graph payload compatible with the existing
`import_youtu_graph` importer so textbook extraction can run fully locally.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


_WS_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")

_DEF_PATTERNS = (
    re.compile(r"(?P<name>[A-Z][A-Za-z0-9 \-]{2,80})\s+is\s+defined\s+as\s+(?P<desc>[^.]{8,220})", re.IGNORECASE),
    re.compile(r"(?P<name>[A-Z][A-Za-z0-9 \-]{2,80})\s+is\s+(?:a|an|the)\s+(?P<desc>[^.]{8,220})", re.IGNORECASE),
    re.compile(r"(?P<name>[\u4e00-\u9fffA-Za-z0-9_]{2,40})\s*是\s*(?P<desc>[^。]{6,120})"),
)

_SKIP_NAMES = {"chapter", "section", "introduction"}


@dataclass(slots=True)
class _Chunk:
    chunk_id: str
    text: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class _Entity:
    entity_id: str
    name: str
    entity_type: str
    description: str
    chunk_id: str
    evidence_quote: str
    char_start: int
    char_end: int
    confidence: float


def _normalize_text(md_text: str) -> str:
    text = str(md_text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Drop headings markers to avoid noisy entity names like "Chapter 1".
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    return text.strip()


def _safe_chunk_id(chapter_id: str, idx: int) -> str:
    base = chapter_id.strip() or "chapter"
    return f"{base}:chunk:{idx:03d}"


def _split_chunks(text: str, chapter_id: str, max_chars: int = 1200) -> list[_Chunk]:
    if not text:
        return []

    paragraphs: list[tuple[str, int, int]] = []
    cursor = 0
    for raw_part in text.split("\n\n"):
        part = raw_part.strip()
        if not part:
            cursor += len(raw_part) + 2
            continue
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        paragraphs.append((part, start, end))
        cursor = end

    chunks: list[_Chunk] = []
    buffer_texts: list[str] = []
    buffer_start: int | None = None
    buffer_end: int | None = None

    def _flush() -> None:
        nonlocal buffer_texts, buffer_start, buffer_end
        if not buffer_texts or buffer_start is None or buffer_end is None:
            return
        idx = len(chunks)
        text_value = "\n\n".join(buffer_texts).strip()
        chunks.append(
            _Chunk(
                chunk_id=_safe_chunk_id(chapter_id, idx),
                text=text_value,
                char_start=buffer_start,
                char_end=buffer_end,
            )
        )
        buffer_texts = []
        buffer_start = None
        buffer_end = None

    for part, start, end in paragraphs:
        candidate = ("\n\n".join(buffer_texts + [part])).strip()
        if buffer_texts and len(candidate) > max_chars:
            _flush()
        if buffer_start is None:
            buffer_start = start
        buffer_end = end
        buffer_texts.append(part)

    _flush()
    return chunks


def _normalize_name(name: str) -> str:
    cleaned = _WS_RE.sub(" ", str(name or "").strip())
    return cleaned.strip(" .,:;!?()[]{}\"'")


def _guess_entity_type(name: str, desc: str) -> str:
    hay = f"{name} {desc}".lower()
    if any(k in hay for k in ("law", "theorem", "principle", "定理", "原理")):
        return "theory"
    if any(k in hay for k in ("method", "algorithm", "方法", "算法")):
        return "method"
    if any(k in hay for k in ("model", "模型")):
        return "model"
    if any(k in hay for k in ("equation", "formula", "方程", "公式")):
        return "equation"
    return "concept"


def _entity_id(name: str) -> str:
    digest = hashlib.sha256(_normalize_name(name).lower().encode("utf-8", errors="ignore")).hexdigest()
    return "local:" + digest[:24]


def _extract_entities(chunks: list[_Chunk]) -> list[_Entity]:
    by_name: dict[str, _Entity] = {}

    for chunk in chunks:
        if not chunk.text:
            continue
        sentences = _SENTENCE_SPLIT_RE.split(chunk.text)
        sentence_cursor = 0
        for sentence in sentences:
            sent = sentence.strip()
            if len(sent) < 12:
                sentence_cursor += len(sentence) + 1
                continue
            sent_local_start = chunk.text.find(sentence, sentence_cursor)
            if sent_local_start < 0:
                sent_local_start = sentence_cursor
            sent_global_start = chunk.char_start + max(0, sent_local_start)
            sentence_cursor = sent_local_start + len(sentence)

            for pattern in _DEF_PATTERNS:
                match = pattern.search(sent)
                if not match:
                    continue
                raw_name = _normalize_name(match.group("name"))
                if not raw_name:
                    continue
                if raw_name.lower() in _SKIP_NAMES:
                    continue
                if len(raw_name) < 3:
                    continue
                desc = _normalize_name(match.group("desc"))
                if len(desc) < 6:
                    continue
                norm = raw_name.lower()
                if norm in by_name:
                    break

                char_start = sent_global_start + match.start("name")
                char_end = sent_global_start + match.end("desc")
                by_name[norm] = _Entity(
                    entity_id=_entity_id(raw_name),
                    name=raw_name,
                    entity_type=_guess_entity_type(raw_name, desc),
                    description=desc,
                    chunk_id=chunk.chunk_id,
                    evidence_quote=sent[:220],
                    char_start=max(0, char_start),
                    char_end=max(0, char_end),
                    confidence=0.72,
                )
                break

    # Keep deterministic order by first appearance in text.
    return sorted(by_name.values(), key=lambda x: (x.char_start, x.name.lower()))


def _build_edges(entities: list[_Entity]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for idx in range(0, max(0, len(entities) - 1)):
        src = entities[idx]
        dst = entities[idx + 1]
        if src.entity_id == dst.entity_id:
            continue
        edge_start = min(src.char_start, dst.char_start)
        edge_end = max(src.char_end, dst.char_end)
        edges.append(
            {
                "start_id": src.entity_id,
                "end_id": dst.entity_id,
                "relation": "related_to",
                "source_chunk_id": src.chunk_id,
                "evidence_quote": f"{src.name} -> {dst.name}",
                "char_start": edge_start,
                "char_end": edge_end,
                "confidence": round(min(src.confidence, dst.confidence) - 0.05, 3),
            }
        )
    return edges


def extract_textbook_graph_local(md_text: str, chapter_id: str) -> dict[str, Any]:
    text = _normalize_text(md_text)
    chunks = _split_chunks(text, chapter_id=chapter_id)
    entities = _extract_entities(chunks)
    edges = _build_edges(entities)

    nodes = [
        {
            "id": entity.entity_id,
            "label": entity.entity_type,
            "properties": {
                "name": entity.name,
                "description": entity.description,
                "source_chunk_id": entity.chunk_id,
                "evidence_quote": entity.evidence_quote,
                "char_start": entity.char_start,
                "char_end": entity.char_end,
                "confidence": entity.confidence,
            },
        }
        for entity in entities
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "communities": [],
    }
