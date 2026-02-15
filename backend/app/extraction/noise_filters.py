"""Noise filtering for extraction pipeline (P0-5).

This module provides filters to detect and remove low-quality claims that should
not be processed as logic claims, including:
- Figure and table captions (e.g., "Figure 1: Experimental setup")
- Pure definitions (future implementation)

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
