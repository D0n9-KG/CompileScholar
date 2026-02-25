from __future__ import annotations

import re
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")

_TYPE_COMPATIBILITY: dict[str, set[str]] = {
    "background": {"concept", "theory", "definition", "principle"},
    "problem": {"concept", "theory", "condition", "definition"},
    "method": {"method", "algorithm", "model", "equation", "concept"},
    "experiment": {"method", "model", "equation", "concept"},
    "result": {"theory", "equation", "model", "principle", "concept"},
    "conclusion": {"theory", "principle", "concept", "model"},
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


def _is_type_compatible(step_type: str, entity_type: str) -> bool:
    s = _normalize_text(step_type)
    e = _normalize_text(entity_type)
    allowed = _TYPE_COMPATIBILITY.get(s)
    if not allowed:
        return True
    return e in allowed


def _compute_link_score(
    step: dict[str, Any],
    entity: dict[str, Any],
    semantic_score: float | None = None,
) -> tuple[float, list[str]]:
    step_summary = str(step.get("summary") or "")
    entity_name = str(entity.get("name") or "")
    entity_desc = str(entity.get("description") or "")
    entity_type = str(entity.get("entity_type") or "concept")
    step_type = str(step.get("step_type") or "")

    step_tokens = _tokenize(step_summary)
    entity_tokens = _tokenize(f"{entity_name} {entity_desc}")
    lexical = _jaccard(step_tokens, entity_tokens)
    semantic = lexical if semantic_score is None else float(max(0.0, min(1.0, semantic_score)))
    compatible = _is_type_compatible(step_type, entity_type)

    # Hard suppression for obvious type mismatch unless lexical overlap is extreme.
    if not compatible and lexical < 0.85:
        return 0.0, [f"type_mismatch:{step_type}->{entity_type}"]

    type_score = 1.0 if compatible else 0.35
    score = 0.50 * lexical + 0.35 * semantic + 0.15 * type_score
    reasons = [
        f"lexical={lexical:.3f}",
        f"semantic={semantic:.3f}",
        f"type={'ok' if compatible else 'soft_mismatch'}",
    ]
    return round(score, 6), reasons


def generate_explains_links(
    logic_steps: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    *,
    min_score: float = 0.45,
    top_k_per_step: int = 3,
    semantic_overrides: dict[tuple[str, str], float] | None = None,
) -> list[dict[str, Any]]:
    semantic_overrides = semantic_overrides or {}
    out: list[dict[str, Any]] = []

    for step in logic_steps:
        step_id = str(step.get("logic_step_id") or "").strip()
        if not step_id:
            continue
        step_links: list[dict[str, Any]] = []
        for entity in entities:
            entity_id = str(entity.get("entity_id") or "").strip()
            if not entity_id:
                continue
            override = semantic_overrides.get((step_id, entity_id))
            score, reasons = _compute_link_score(step, entity, semantic_score=override)
            if score < float(min_score):
                continue
            step_links.append(
                {
                    "logic_step_id": step_id,
                    "paper_id": str(step.get("paper_id") or ""),
                    "entity_id": entity_id,
                    "source_chapter_id": str(entity.get("source_chapter_id") or ""),
                    "score": score,
                    "reasons": reasons,
                    "evidence_chunk_ids": list(step.get("evidence_chunk_ids") or []),
                }
            )

        step_links.sort(key=lambda x: (-float(x.get("score") or 0.0), str(x.get("entity_id") or "")))
        out.extend(step_links[: max(1, int(top_k_per_step))])

    out.sort(key=lambda x: (str(x.get("logic_step_id") or ""), -float(x.get("score") or 0.0), str(x.get("entity_id") or "")))
    return out
