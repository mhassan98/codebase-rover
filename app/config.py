from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.indexing.limits import EMBEDDING_DIM

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://rover:rover@localhost:5432/rover"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
    embedding_dim: int = EMBEDDING_DIM
    hosted_model_enabled: bool = False
    hosted_model_base_url: str = ""
    hosted_model_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    hosted_model_name: str = ""
    rover_allowed_roots: str = ""
    rover_config: str = ""
    log_level: str = "INFO"
    max_diff_chars: int = 500_000
    max_context_chunks: int = 8
    max_retrieved_chars: int = 24_000
    max_diff_prompt_chars: int = 80_000
    max_prompt_chars: int = 120_000
    chat_timeout_seconds: float = 180
    embed_timeout_seconds: float = 30

    def allowed_roots(self) -> list[str]:
        raw = self.rover_allowed_roots.strip()
        if not raw:
            return []
        return [part for part in raw.split(os.pathsep) if part]


@lru_cache
def get_settings() -> Settings:
    return Settings()
