from __future__ import annotations

import re
from typing import Any


SUPPORT_MARKERS = [
    "support",
    "consistent with",
    "agree with",
    "confirm",
    "corroborate",
    "证实",
    "一致",
    "支持",
]
CHALLENGE_MARKERS = [
    "however",
    "but",
    "not",
    "fails to",
    "cannot",
    "limitation",
    "inconsistent",
    "contrary",
    "conflict",
    "challenge",
    "不足",
    "局限",
    "并不",
    "不一致",
    "冲突",
    "挑战",
]
SUPERSEDE_MARKERS = [
    "outperform",
    "better than",
    "surpass",
    "replace",
    "supersede",
    "state-of-the-art",
    "sota",
    "significantly improve",
    "优于",
    "超过",
    "替代",
    "更好",
    "显著提升",
]


_WS_RE = re.compile(r"\s+")


def normalize_proposition_text(text: str) -> str:
    s = _WS_RE.sub(" ", (text or "").strip().lower())
    while s and s[-1] in ".;銆傦紱":
        s = s[:-1].rstrip()
    return s


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def contains_any(text: str, markers: list[str]) -> bool:
    t = text or ""
    return any(m in t for m in markers)


def _best_purpose_score(labels: list[str] | None, scores: list[float] | None, wanted: str) -> float:
    """Return the highest score among citations with the given purpose label."""
    labs = [str(x).strip() for x in (labels or []) if str(x).strip()]
    vals = list(scores or [])
    best = 0.0
    for idx, label in enumerate(labs):
        if label != wanted:
            continue
        try:
            raw = float(vals[idx]) if idx < len(vals) else 0.4
        except Exception:
            raw = 0.4
        best = max(best, clamp01(raw))
    return best


def infer_relation_type(
    source_text: str,
    target_text: str,
    similarity: float,
    target_confidence: float,
    *,
    citation_purpose_labels: list[str] | None = None,
    citation_purpose_scores: list[float] | None = None,
    min_similarity: float = 0.86,
    accepted_threshold: float = 0.82,
) -> dict[str, Any] | None:
    sim = clamp01(similarity)
    tgt_conf = clamp01(target_confidence)
    if sim < min_similarity:
        return None

    src = normalize_proposition_text(source_text)
    tgt = normalize_proposition_text(target_text)
    if not src or not tgt:
        return None

    # Text identity detection: identical normalized text should trigger merge, not relation
    if src == tgt:
        # High confidence merge signal (not a relation type)
        return {"event_type": "MERGE", "confidence": 0.99, "strength": 0.99, "status": "accepted", "reason": "text_identity"}

    base_conf = clamp01(0.65 * sim + 0.35 * tgt_conf)

    # Note: removed "if src == tgt and sim >= 0.96:" block - now handled above as MERGE

    supersedes = contains_any(tgt, SUPERSEDE_MARKERS)
    challenges = contains_any(tgt, CHALLENGE_MARKERS)
    supports = contains_any(tgt, SUPPORT_MARKERS)

    # Citation purpose scores from the LLM-extracted citation context.
    # These are more reliable signals than keyword matching in proposition text.
    p_supersede = _best_purpose_score(citation_purpose_labels, citation_purpose_scores, "ExtendImprove")
    p_challenge = _best_purpose_score(citation_purpose_labels, citation_purpose_scores, "CritiqueLimit")
    p_support = _best_purpose_score(citation_purpose_labels, citation_purpose_scores, "SupportEvidence")

    # SUPERSEDES: keyword match at high similarity OR strong ExtendImprove purpose signal
    if (supersedes and sim >= 0.90) or (p_supersede >= 0.60 and sim >= 0.86):
        conf = clamp01(base_conf + 0.06 + 0.05 * p_supersede)
        status = "accepted" if conf >= accepted_threshold else "pending_review"
        return {"event_type": "SUPERSEDES", "confidence": conf, "strength": conf, "status": status}

    # CHALLENGES: keyword match at high similarity OR strong CritiqueLimit purpose signal
    if (challenges and sim >= 0.90) or (p_challenge >= 0.55 and sim >= 0.86):
        conf = clamp01(base_conf + 0.04 + 0.06 * p_challenge)
        status = "accepted" if conf >= accepted_threshold else "pending_review"
        return {"event_type": "CHALLENGES", "confidence": conf, "strength": conf, "status": status}

    # SUPPORTS: keyword match OR SupportEvidence purpose signal
    if (supports and sim >= 0.89) or (p_support >= 0.50 and sim >= 0.88):
        conf = clamp01(base_conf + 0.03 + 0.04 * p_support)
        status = "accepted" if conf >= accepted_threshold else "pending_review"
        return {"event_type": "SUPPORTS", "confidence": conf, "strength": conf, "status": status}

    if sim >= 0.97:
        conf = clamp01(base_conf)
        status = "accepted" if conf >= accepted_threshold else "pending_review"
        return {"event_type": "SUPPORTS", "confidence": conf, "strength": conf, "status": status}

    return None
