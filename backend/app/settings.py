from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Prefer backend/.env, but also allow repo-root .env (common in this project).
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str = Field(default="bolt://localhost:7687", validation_alias=AliasChoices("NEO4J_URI"))
    neo4j_user: str = Field(default="neo4j", validation_alias=AliasChoices("NEO4J_USER", "NEO4J_USERNAME"))
    neo4j_password: str = Field(default="neo4j_password", validation_alias=AliasChoices("NEO4J_PASSWORD"))

    # LLM config (default: DeepSeek via OpenAI-compatible API).
    llm_provider: str = Field(
        default="deepseek",
        description="deepseek | openrouter | openai",
        validation_alias=AliasChoices("LLM_PROVIDER"),
    )
    llm_base_url: str | None = Field(default=None, validation_alias=AliasChoices("LLM_BASE_URL"))
    llm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("LLM_API_KEY"))
    llm_model: str = Field(default="deepseek-chat", validation_alias=AliasChoices("LLM_MODEL"))

    # Keys (read from env). We'll select based on llm_provider unless llm_api_key is set.
    deepseek_api_key: str | None = Field(default=None, validation_alias=AliasChoices("DEEPSEEK_API_KEY", "DEEPSEEK_KEY"))
    openrouter_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY", "OPENROUTER_KEY")
    )
    openai_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_API_KEY"))

    # Embeddings (optional). If not available, system falls back to lexical retrieval.
    embedding_provider: str | None = Field(
        default=None,
        description="siliconflow | openai | openrouter | deepseek | (None=disable)",
        validation_alias=AliasChoices("EMBEDDING_PROVIDER"),
    )
    embedding_base_url: str | None = Field(default=None, validation_alias=AliasChoices("EMBEDDING_BASE_URL"))
    embedding_api_key: str | None = Field(default=None, validation_alias=AliasChoices("EMBEDDING_API_KEY"))
    embedding_model: str | None = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("EMBEDDING_MODEL"),
    )

    siliconflow_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SILICONFLOW_API_KEY", "SILICON_FLOW_API_KEY", "SILICONCLOUD_API_KEY"),
    )

    data_root: str = ".."
    storage_dir: str = "storage"

    def effective_llm_api_key(self) -> str | None:
        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key or self.openai_api_key
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key or self.openai_api_key
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.openai_api_key or self.deepseek_api_key or self.openrouter_api_key

    def effective_llm_base_url(self) -> str | None:
        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider == "deepseek":
            return "https://api.deepseek.com/v1"
        if self.llm_provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        return None

    def effective_embedding_api_key(self) -> str | None:
        if self.embedding_api_key:
            return self.embedding_api_key
        provider = self.effective_embedding_provider()
        if provider == "siliconflow":
            return self.siliconflow_api_key
        if provider == "openrouter":
            return self.openrouter_api_key or self.openai_api_key
        if provider == "deepseek":
            return self.deepseek_api_key or self.openai_api_key
        if provider == "openai":
            return self.openai_api_key
        # default: try whatever exists
        return self.openai_api_key or self.siliconflow_api_key or self.openrouter_api_key or self.deepseek_api_key

    def effective_embedding_base_url(self) -> str | None:
        if self.embedding_base_url:
            return self.embedding_base_url
        provider = self.effective_embedding_provider()
        if provider == "siliconflow":
            # SiliconFlow publishes both .cn and .com endpoints; default to .cn but allow override via EMBEDDING_BASE_URL.
            return "https://api.siliconflow.cn/v1"
        if provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        if provider == "deepseek":
            return "https://api.deepseek.com/v1"
        return None

    def effective_embedding_provider(self) -> str:
        explicit = (self.embedding_provider or "").lower().strip()
        if explicit:
            return explicit
        # inference: if a dedicated key is present, assume that provider
        if self.siliconflow_api_key:
            return "siliconflow"
        if self.openai_api_key:
            return "openai"
        if self.openrouter_api_key:
            return "openrouter"
        if self.deepseek_api_key:
            return "deepseek"
        return ""

    def effective_embedding_model(self) -> str | None:
        provider = self.effective_embedding_provider()
        if provider == "siliconflow":
            # If user didn't explicitly set a model, prefer BGE-M3 as requested.
            if self.embedding_model and self.embedding_model != "text-embedding-3-small":
                return self.embedding_model
            return "BAAI/bge-m3"
        return self.embedding_model


settings = Settings()
