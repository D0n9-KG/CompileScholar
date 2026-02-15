from __future__ import annotations

import json
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.settings import settings


_JSON_BLOCK_RE = re.compile(r"```json\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    m = _JSON_BLOCK_RE.search(text)
    if m:
        text = m.group("body").strip()
    # Try to locate first { ... } if extra text exists
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def llm() -> Any:
    from langchain_openai import ChatOpenAI

    api_key = settings.effective_llm_api_key()
    base_url = settings.effective_llm_base_url()
    if not api_key:
        raise RuntimeError("LLM API key missing (set DEEPSEEK_API_KEY or LLM_API_KEY)")
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=settings.llm_model,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=0.8, max=4.0))
def _call_text_with_retry(system: str, user: str) -> str:
    """Call LLM with retry logic and return raw text response."""
    resp = llm().invoke([("system", system), ("user", user)])
    return str(resp.content or "")


def call_text(system: str, user: str, *, use_retry: bool = True) -> str:
    """
    Call LLM and return raw text response.

    Args:
        system: System prompt
        user: User prompt
        use_retry: Whether to use retry logic (default: True)

    Returns:
        Raw text response from LLM
    """
    if use_retry:
        return _call_text_with_retry(system, user)
    resp = llm().invoke([("system", system), ("user", user)])
    return str(resp.content or "")


def call_json(system: str, user: str, *, use_retry: bool = True) -> dict:
    """
    Call LLM and parse JSON response.

    Args:
        system: System prompt
        user: User prompt
        use_retry: Whether to use retry logic (default: True)

    Returns:
        Parsed JSON dict

    Raises:
        JSONDecodeError: If response is not valid JSON
    """
    raw = call_text(system, user, use_retry=use_retry)
    return _extract_json(raw)

