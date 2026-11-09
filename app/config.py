from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings. Read from environment variables, or from .env when run outside Docker."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "info"

    # Uploaded PDFs, page images, SQLite and ChromaDB all live under this folder.
    data_dir: Path = Path("data")

    # Ollama on this machine (reachable from the container thanks to host networking).
    ollama_base_url: str = "http://127.0.0.1:11434"
    # How long a model stays in memory after its last use.
    ollama_keep_alive: str = "10m"


@lru_cache
def get_settings() -> Settings:
    return Settings()
