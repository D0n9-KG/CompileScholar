from __future__ import annotations

import re
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")

_ROLE_COMPATIBILITY: dict[str, set[str]] = {
    "background": {"concept", "theory", "definition", "principle"},
    "problem": {"concept", "theory", "condition", "definition"},
    "method": {"method", "algorithm", "model", "equation", "concept"},
    "experiment": {"method", "model", "equation", "concept"},
    "result": {"theory", "equation", "model", "principle", "concept"},
    "interpretation": {"theory", "principle", "concept", "model"},
    "limitation": {"concept", "condition", "principle"},
    "future_work": {"concept", "method", "model", "condition"},
}

_GENERIC_ENTITY_TYPES = {
    "",
    "entity",
    "term",
    "knowledgeentity",
    "knowledge_entity",
    "conceptual_entity",
}


def _normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip().lower())


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(str(text or ""))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    union = len(a | b)
    return inter / max(1, union)


def _is_type_compatible(role: str, entity_type: str) -> bool:
    s = _normalize_text(role)
    e = _normalize_text(entity_type)
    if e in _GENERIC_ENTITY_TYPES:
        return True
    allowed = _ROLE_COMPATIBILITY.get(s)
    if not allowed:
        return True
    return e in allowed


def _compute_link_score(
    move: dict[str, Any],
    entity: dict[str, Any],
    semantic_score: float | None = None,
) -> tuple[float, list[str]]:
    move_summary = str(move.get("summary") or "")
    entity_name = str(entity.get("name") or "")
    entity_desc = str(entity.get("description") or "")
    entity_type = str(entity.get("entity_type") or "concept")
    role = str(move.get("role") or "")

    move_tokens = _tokenize(move_summary)
    entity_tokens = _tokenize(f"{entity_name} {entity_desc}")
    lexical = _jaccard(move_tokens, entity_tokens)
    overlap = len(move_tokens & entity_tokens)
    entity_coverage = overlap / max(1, len(entity_tokens))
    name_norm = _normalize_text(entity_name)
    summary_norm = _normalize_text(move_summary)
    name_in_summary = bool(name_norm and len(name_norm) >= 4 and name_norm in summary_norm)

    if semantic_score is None:
        semantic = min(1.0, 0.6 * lexical + 0.4 * entity_coverage)
    else:
        semantic = float(max(0.0, min(1.0, semantic_score)))

    compatible = _is_type_compatible(role, entity_type)

    # Hard suppression for obvious type mismatch with no textual evidence.
    if not compatible and overlap <= 0 and not name_in_summary:
        return 0.0, [f"type_mismatch:{role}->{entity_type}"]

    type_score = 1.0 if compatible else 0.35
    name_hit = 1.0 if name_in_summary else 0.0
    score = 0.25 * lexical + 0.30 * entity_coverage + 0.20 * semantic + 0.10 * name_hit + 0.15 * type_score
    reasons = [
        f"lexical={lexical:.3f}",
        f"coverage={entity_coverage:.3f}",
        f"semantic={semantic:.3f}",
        f"name_hit={'1' if name_in_summary else '0'}",
        f"type={'ok' if compatible else 'soft_mismatch'}",
    ]
    return round(score, 6), reasons


def generate_explains_links(
    research_moves: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    *,
    min_score: float = 0.45,
    top_k_per_move: int = 3,
    semantic_overrides: dict[tuple[str, str], float] | None = None,
) -> list[dict[str, Any]]:
    semantic_overrides = semantic_overrides or {}
    out: list[dict[str, Any]] = []

    for move in research_moves:
        move_id = str(move.get("move_id") or "").strip()
        if not move_id:
            continue
        move_links: list[dict[str, Any]] = []
        for entity in entities:
            entity_id = str(entity.get("entity_id") or "").strip()
            if not entity_id:
                continue
            override = semantic_overrides.get((move_id, entity_id))
            score, reasons = _compute_link_score(move, entity, semantic_score=override)
            if score < float(min_score):
                continue
            move_links.append(
                {
                    "move_id": move_id,
                    "paper_id": str(move.get("paper_id") or ""),
                    "paper_source": str(move.get("paper_source") or ""),
                    "role": str(move.get("role") or ""),
                    "act_type": str(move.get("act_type") or ""),
                    "summary": str(move.get("summary") or ""),
                    "entity_id": entity_id,
                    "source_chapter_id": str(entity.get("source_chapter_id") or ""),
                    "score": score,
                    "reasons": reasons,
                    "anchor_ids": [str(item).strip() for item in (move.get("anchor_ids") or []) if str(item).strip()],
                    "evidence_quote": f"{str(move.get('summary') or '')[:120]} | {str(entity.get('name') or '')}",
                }
            )

        move_links.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("entity_id") or "")))
        out.extend(move_links[: max(1, int(top_k_per_move))])

    out.sort(key=lambda x: (str(x.get("move_id") or ""), -float(x.get("score") or 0.0), str(x.get("entity_id") or "")))
    return out
