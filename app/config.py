from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    # Vision step: who reads charts and scanned pages. "local" = Qwen2.5-VL in
    # Ollama, "aws" = Nova Lite, "none" = only the PDF's own text.
    default_extractor: Literal["local", "aws", "none"] = "local"
    vision_local_model: str = "qwen2.5vl:latest"
    vision_num_ctx: int = 8192  # room for one image plus the reply
    vision_max_megapixels: float = 1.0  # figures with more pixels than this are scaled down first
    vision_aws_concurrency: int = 4

    # Answers
    chat_num_ctx: int = 8192  # model context for a local chat model (same as the vision model's, so no reload)
    chat_max_sources: int = 8  # how many search results the model reads
    chat_max_source_chars: int = 12000  # total size of those results (about 3,000 tokens)
    chat_max_images: int = 2  # chart images attached when the model can read images
    chat_max_answer_tokens: int = 1024

    # Ollama on this machine (reachable from the container thanks to host networking).
    ollama_base_url: str = "http://127.0.0.1:11434"
    # How long a model stays in memory after its last use.
    ollama_keep_alive: str = "10m"

    # The search model runs inside the app, on the CPU. Its files are built into the image.
    embed_model_dir: Path = Path(__file__).resolve().parent.parent / "models" / "nomic-embed-text-v1.5"
    embed_threads: int = 0  # CPU threads for the search model; 0 = up to 4
    embed_batch_size: int = 32  # chunks embedded between progress updates

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
