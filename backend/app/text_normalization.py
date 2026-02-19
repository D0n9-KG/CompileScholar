"""Text normalization utilities for encoding recovery and symbol folding."""
from __future__ import annotations

# Markers commonly seen when UTF-8 text is mis-decoded as GBK/CP936.
# We keep this conservative and only use it as a trigger heuristic.
_MOJIBAKE_MARKERS = (
    "寮曡█",   # 引言
    "鎽樿",   # 摘要
    "鍙傝€冩枃鐚",  # 参考文献
    "鍩",       # often appears in mojibake Chinese headings
    "銆",       # malformed punctuation cluster
    "锛",
    "锟",
    "鈥",
)

# Common symbol confusables observed in claims/chunks.
# Keep 1:1 replacements so index-based matching remains stable.
_SYMBOL_CONFUSABLES = str.maketrans(
    {
        "胃": "θ",
        "惟": "Ω",
        "伪": "α",
        "尾": "β",
        "纬": "γ",
        "渭": "μ",
        "蟽": "σ",
        "掳": "°",
    }
)


def _marker_score(text: str) -> int:
    s = text or ""
    return sum(s.count(token) for token in _MOJIBAKE_MARKERS)


def maybe_recover_utf8_as_gbk_mojibake(text: str) -> str:
    """Recover text that was likely decoded as GBK from UTF-8 bytes.

    Strategy:
    - only attempt recovery when mojibake markers are sufficiently frequent
    - reverse by gb18030-encode -> utf-8-decode
    - accept candidate only if marker score is clearly improved
    """
    raw = text or ""
    raw_score = _marker_score(raw)
    if raw_score < 3:
        return raw

    try:
        candidate = raw.encode("gb18030", errors="strict").decode("utf-8", errors="strict")
    except UnicodeError:
        return raw

    if not candidate:
        return raw

    cand_score = _marker_score(candidate)
    if cand_score + 1 <= raw_score:
        return candidate
    return raw


def normalize_ingested_markdown(text: str) -> str:
    """Ingestion-stage normalization:
    - recover UTF-8/GBK mojibake when confidently detected
    """
    return maybe_recover_utf8_as_gbk_mojibake(text or "")


def fold_symbol_confusables(text: str) -> str:
    """Span-matching-only symbol folding (no semantic rewrite)."""
    return (text or "").translate(_SYMBOL_CONFUSABLES)
