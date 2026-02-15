"""Noise filtering for extraction pipeline (P0-5).

This module provides filters to detect and remove low-quality claims that should
not be processed as logic claims, including:
- Figure and table captions (e.g., "Figure 1: Experimental setup")
- Pure definitions (e.g., "X is a Y", "X refers to Y")

These filters improve extraction quality by preventing the LLM from spending
tokens on non-substantive text.
"""
from __future__ import annotations

import re


# Pattern: "Figure 1:", "Table 12:", "Fig. 3:"
# Note: Phase 1 focuses on standard captions. Future expansions may include:
#   - Supplementary figures: "Supplementary Figure 1:", "Figure S1:"
#   - Subfigures: "Figure 1A:", "Figure 1a:"
#   - Appendix figures: "Figure A1:"
_CAPTION_PATTERN = re.compile(
    r"^\s*(Figure|Table|Fig\.)\s+\d+\s*:",
    re.IGNORECASE
)


# Definition patterns: "X is a Y", "X refers to Y", "X is defined as Y"
_DEFINITION_PATTERNS = [
    re.compile(r"\bis\s+a\s+", re.IGNORECASE),  # "is a"
    re.compile(r"\bis\s+the\s+", re.IGNORECASE),  # "is the"
    re.compile(r"\brefers?\s+to\s+", re.IGNORECASE),  # "refers to"
    re.compile(r"\bis\s+defined\s+as\s+", re.IGNORECASE),  # "is defined as"
    re.compile(r"\brepresents?\s+", re.IGNORECASE),  # "represents"
]

# Comparative/causal patterns (using \w* to catch inflections)
# Note: "lead" requires suffix to avoid matching "leadership"
_COMPARATIVE_PATTERN = re.compile(
    r'\b(better|worse|more|less|higher|lower|'
    r'outperform\w*|improv\w+|increas\w+|decreas\w+|'
    r'caus\w+|leads?|leading|led|result\w*)\b',
    re.IGNORECASE
)

# Strong definition markers that don't require is/are density check
_STRONG_DEFINITION_MARKERS = [
    re.compile(r"\brefers?\s+to\s+", re.IGNORECASE),
    re.compile(r"\bis\s+defined\s+as\s+", re.IGNORECASE),
]


def is_caption_text(text: str | None) -> bool:
    """Detect if text appears to be a figure or table caption.

    This function identifies caption-like text that starts with common
    scientific figure/table patterns. It is intentionally conservative
    and will NOT match:
    - Figure references in sentences ("as shown in Figure 1")
    - Captions without colons ("Figure 1 shows...")
    - Variants without dots ("Fig 1:")

    Args:
        text: Claim text to check. Can be None.

    Returns:
        True if text starts with a figure/table caption pattern,
        False otherwise (including for None/empty inputs).

    Examples:
        >>> is_caption_text("Figure 1: Experimental setup")
        True
        >>> is_caption_text("Table 5: Summary statistics")
        True
        >>> is_caption_text("Fig. 3: Data distribution")
        True
        >>> is_caption_text("as shown in Figure 1")
        False
        >>> is_caption_text("This demonstrates the method")
        False
        >>> is_caption_text(None)
        False
    """
    if not isinstance(text, str):
        return False

    # Pattern already handles leading whitespace via ^\s*
    return _CAPTION_PATTERN.match(text) is not None


def is_pure_definition_text(text: str | None) -> bool:
    """Detect if text is a pure definition.

    Pure definitions have low semantic value for evolution relations.
    Examples: "X is a Y", "X refers to Y", "X is defined as Y"

    Heuristic:
    - Definition if (pattern_count >= 1 AND is/are density > 0.08, ~1 copula per 12 words)
      OR pattern_count >= 2
      OR contains strong definition markers ("refers to", "defined as")
    - Rejects comparative/causal statements (e.g., "outperforms", "causes")

    Args:
        text: Claim text to check. Can be None.

    Returns:
        True if text appears to be a pure definition

    Examples:
        >>> is_pure_definition_text("Machine learning is a method of data analysis")
        True
        >>> is_pure_definition_text("This term refers to the process of optimization")
        True
        >>> is_pure_definition_text("This approach outperforms previous methods")
        False
    """
    if not isinstance(text, str):
        return False

    text_lower = text.lower()

    # Reject comparative/causal statements (these are substantive claims)
    if _COMPARATIVE_PATTERN.search(text_lower):
        return False

    # Strong definition markers that don't require is/are density check
    if any(marker.search(text_lower) for marker in _STRONG_DEFINITION_MARKERS):
        return True

    # Count all definition patterns
    pattern_count = sum(1 for pattern in _DEFINITION_PATTERNS if pattern.search(text_lower))

    # Check for high is/are density (definition marker)
    is_are_count = len(re.findall(r'\b(is|are|was|were)\b', text_lower))
    total_words = len(text.split())

    if total_words == 0:
        return False

    is_are_density = is_are_count / total_words

    # Heuristic: definition if has pattern AND high is/are density
    # OR multiple definition patterns
    return (pattern_count >= 1 and is_are_density > 0.08) or pattern_count >= 2
