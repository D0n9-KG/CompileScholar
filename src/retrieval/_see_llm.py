from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI


PROVIDER_ENV = (
    ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL"),
    ("peratera", "PERATERA_API_KEY", "PERATERA_BASE_URL"),
    ("cst", "CST_API_KEY", "CST_BASE_URL"),
)
MODEL_ENV = ("SCIEVO_LLM_MODEL", "LLM_MODEL", "RAGANYTHING_LLM_MODEL")


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    api_key: str
    base_url: str | None
    model: str

    def trace_metadata(self) -> dict[str, str | None]:
        return {
            "provider": self.provider,
            "base_url_host": _host(self.base_url),
            "model": self.model,
        }


def load_llm_config(env_file: Path | None = None) -> LlmConfig | None:
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    model = _first_env(MODEL_ENV) or "gpt-4.1-mini"
    for provider, key_name, base_url_name in PROVIDER_ENV:
        api_key = os.getenv(key_name)
        if api_key:
            return LlmConfig(
                provider=provider,
                api_key=api_key,
                base_url=os.getenv(base_url_name),
                model=model,
            )
    return None


def create_openai_client(config: LlmConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _host(url: str | None) -> str | None:
    if not url:
        return None
    return urlparse(url).netloc or url
