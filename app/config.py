from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings. Read from environment variables, or from .env when run outside Docker."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "info"

    # Uploaded PDFs, page images, SQLite and ChromaDB all live under this folder.
    data_dir: Path = Path("data")

    # PDF uploads
    max_upload_mb: int = 100
    page_preview_dpi: int = 100  # page images shown in the UI
    figure_dpi: int = 200  # cropped charts and pictures sent to the vision model

    # Ollama on this machine (reachable from the container thanks to host networking).
    ollama_base_url: str = "http://127.0.0.1:11434"
    # How long a model stays in memory after its last use.
    ollama_keep_alive: str = "10m"
    # Embedding model for search. Runs on CPU and is never unloaded.
    ollama_embed_model: str = "nomic-embed-text"
    embed_batch_size: int = 32

    # AWS Bedrock. Keys can also come from ~/.aws or any other standard AWS source.
    aws_region: str = "us-east-1"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    aws_session_token: SecretStr | None = None
    bedrock_nova_lite_model_id: str = "amazon.nova-lite-v1:0"
    # ARN of Qwen2.5-VL imported with Bedrock Custom Model Import. Empty = not used.
    bedrock_qwen_vl_model_arn: str = ""
    # Imported models sleep when idle; how long to wait for one to wake up (seconds).
    bedrock_cold_start_wait: int = 300

    def secret(self, name: str) -> str | None:
        value = getattr(self, name)
        return value.get_secret_value() if value else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
