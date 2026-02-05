from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI
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


def llm() -> ChatOpenAI:
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
def call_json(system: str, user: str) -> dict:
    resp = llm().invoke([("system", system), ("user", user)])
    return _extract_json(resp.content)

