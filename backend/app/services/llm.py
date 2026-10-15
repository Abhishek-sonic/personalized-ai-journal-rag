"""LLM abstraction over any OpenAI-compatible chat API.

Both Ollama and OpenRouter implement the OpenAI `/chat/completions` spec, so
one thin wrapper around the official SDK covers both. Switching provider is
a .env change only (LLM_PROVIDER); nothing here branches on provider except
the optional OpenRouter attribution headers.
"""
import logging
from collections.abc import Iterator
from functools import lru_cache

from openai import APIConnectionError, OpenAI

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """The provider could not be reached. The API layer turns this into a 502."""


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, provider: str, extra_headers: dict | None = None):
        self.provider = provider
        self.base_url = base_url
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, default_headers=extra_headers)

    def _unavailable(self) -> LLMUnavailableError:
        hint = "Is Ollama running? Try `ollama serve`." if self.provider == "ollama" else "Check your network and API key."
        return LLMUnavailableError(f"Cannot reach {self.provider} at {self.base_url}. {hint}")

    def complete(self, messages: list[dict]) -> str:
        try:
            response = self._client.chat.completions.create(model=self.model, messages=messages, temperature=0.2)
        except APIConnectionError as exc:
            raise self._unavailable() from exc
        return response.choices[0].message.content or ""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        try:
            chunks = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2, stream=True
            )
            for chunk in chunks:
                # Some providers send keep-alive chunks with no choices or no text.
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APIConnectionError as exc:
            raise self._unavailable() from exc


def get_llm_client(settings: Settings) -> LLMClient:
    """Factory: pick base URL, key and model from LLM_PROVIDER via Settings."""
    extra_headers = None
    if settings.llm_provider == "openrouter":
        # Optional headers OpenRouter uses to attribute requests to this app.
        extra_headers = {"HTTP-Referer": settings.openrouter_referer, "X-Title": settings.openrouter_title}
    client = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        provider=settings.llm_provider,
        extra_headers=extra_headers,
    )
    # Never log the API key.
    log.info("[LLM] provider=%s base_url=%s model=%s", client.provider, client.base_url, client.model)
    return client


@lru_cache
def get_llm() -> LLMClient:
    # Cached so the HTTP connection pool is reused across requests.
    return get_llm_client(get_settings())
