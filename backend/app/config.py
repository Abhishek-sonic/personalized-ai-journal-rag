"""Single source of environment configuration.

Every URL, model name and tunable lives here (as a default) or in .env.
No other module reads os.environ or hard-codes a provider detail.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env sits at the repo root, one level above backend/, so uvicorn works
# no matter which directory it is started from.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # --- Supabase ---
    supabase_url: str
    supabase_anon_key: str
    # Legacy HS256 projects only. Empty -> verify tokens through the JWKS endpoint instead.
    supabase_jwt_secret: str = ""

    # --- LLM provider toggle ---
    llm_provider: Literal["ollama", "openrouter"] = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-3.5-haiku"
    # Optional headers OpenRouter uses to attribute traffic to an app.
    openrouter_referer: str = "http://localhost:8000"
    openrouter_title: str = "AI Journal"

    # --- Embeddings (independent of the chat provider) ---
    embedding_provider: Literal["fastembed", "openai_compatible"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # must equal vector(N) in supabase/schema.sql
    embedding_base_url: str = ""
    embedding_api_key: str = ""

    # --- RAG tuning ---
    rag_top_k: int = 5
    rag_min_similarity: float = 0.3
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_provider_requirements(self) -> "Settings":
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY")
        if self.embedding_provider == "openai_compatible" and not self.embedding_base_url:
            raise ValueError("EMBEDDING_PROVIDER=openai_compatible requires EMBEDDING_BASE_URL")
        return self

    # Effective LLM settings: the factory in services/llm.py only reads these three,
    # so switching provider is purely a .env change.
    @property
    def llm_base_url(self) -> str:
        return self.ollama_base_url if self.llm_provider == "ollama" else self.openrouter_base_url

    @property
    def llm_api_key(self) -> str:
        # Ollama ignores the key but the OpenAI SDK refuses an empty one.
        return "ollama" if self.llm_provider == "ollama" else self.openrouter_api_key

    @property
    def llm_model(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.openrouter_model

    @property
    def jwks_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


@lru_cache
def get_settings() -> Settings:
    # Cached so .env is parsed once; also handy to override in tests.
    return Settings()
