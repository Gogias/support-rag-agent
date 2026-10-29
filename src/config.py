"""Адреса сервисов и имя коллекции. Значения по умолчанию — локальный стек; переопределяются в .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # .env из корня проекта, откуда бы ни запускали; остальные ключи .env (Langfuse, GigaChat) пропускаем
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "http://localhost:5001/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "qwen3.5"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "MTS_support_v2"

    embed_url: str = "http://localhost:5002/embedding"
    rerank_url: str = "http://127.0.0.1:5003/v1/rerank"


settings = Settings()