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


def infer_relation_type(
    source_text: str,
    target_text: str,
    similarity: float,
    target_confidence: float,
    *,
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
    status = "pending_review"

    # Note: removed "if src == tgt and sim >= 0.96:" block - now handled above as MERGE

    supersedes = contains_any(tgt, SUPERSEDE_MARKERS)
    challenges = contains_any(tgt, CHALLENGE_MARKERS)
    supports = contains_any(tgt, SUPPORT_MARKERS)

    if supersedes and sim >= 0.90:
        conf = clamp01(base_conf + 0.06)
        status = "accepted" if conf >= accepted_threshold else "pending_review"
        return {"event_type": "SUPERSEDES", "confidence": conf, "strength": conf, "status": status}

    if challenges and sim >= 0.90:
        conf = clamp01(base_conf + 0.04)
        status = "accepted" if conf >= accepted_threshold else "pending_review"
        return {"event_type": "CHALLENGES", "confidence": conf, "strength": conf, "status": status}

    if supports and sim >= 0.89:
        conf = clamp01(base_conf + 0.03)
        status = "accepted" if conf >= accepted_threshold else "pending_review"
        return {"event_type": "SUPPORTS", "confidence": conf, "strength": conf, "status": status}

    if sim >= 0.97:
        conf = clamp01(base_conf)
        return {"event_type": "SUPPORTS", "confidence": conf, "strength": conf, "status": "pending_review"}

    return None
