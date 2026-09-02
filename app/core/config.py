"""Typed application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SUPPORTED_AZURE_OPENAI_AUTH_MODES = {"api_key", "entra"}


def normalize_azure_openai_v1_base_url(value: str) -> str:
    """Validate and normalize a user-supplied OpenAI-compatible Azure v1 URL."""

    candidate = value.strip()
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("AZURE_OPENAI_BASE_URL must be an absolute HTTPS URL ending in /v1")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("AZURE_OPENAI_BASE_URL must not contain credentials, a query, or a fragment")
    if parsed.hostname.lower() == "api.openai.com":
        raise ValueError("Public OpenAI endpoints are not permitted")
    normalized_path = parsed.path.rstrip("/")
    if not normalized_path.lower().endswith("/v1"):
        raise ValueError("AZURE_OPENAI_BASE_URL must include an OpenAI-compatible /v1 path")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), f"{normalized_path}/", "", ""))


class Settings(BaseSettings):
    """Runtime settings with safe defaults for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Parse Before You Prompt"
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8080, ge=1, le=65535)
    ui_api_base_url: str = "http://127.0.0.1:8080"

    database_url: str = (
        "postgresql+psycopg://parse_prompt:parse_prompt@localhost:5432/parse_prompt"
    )
    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = False

    upload_root: Path = Path("./data/uploads")
    artifact_root: Path = Path("./data/artifacts")
    hf_home: Path = Path("./data/model-cache/huggingface")
    docling_artifacts_path: Path | None = None
    docling_device: str = "auto"
    docling_num_threads: int = Field(default=4, ge=1)
    docling_max_file_size_mb: int = Field(default=25, ge=1)
    docling_max_pages: int = Field(default=30, ge=1)
    docling_image_scale: float = Field(default=2.0, gt=0)
    docling_enable_picture_description: bool = True
    docling_enable_formula_enrichment: bool = True
    docling_enable_code_enrichment: bool = True
    docling_enable_chart_extraction: bool = False
    docling_enable_vlm_comparison: bool = True

    chunk_max_tokens: int = Field(default=800, ge=1)
    baseline_chunk_overlap_tokens: int = Field(default=100, ge=0)
    chunk_merge_peers: bool = True
    chunk_repeat_table_header: bool = True
    chunk_omit_header_on_overflow: bool = False
    tiktoken_model_hint: str = "text-embedding-3-large"
    tiktoken_fallback_encoding: str = "cl100k_base"

    embedding_batch_max_inputs: int = Field(default=16, ge=1)
    embedding_batch_max_tokens: int = Field(default=8000, ge=1)

    rag_top_k: int = Field(default=5, ge=1)
    rag_max_top_k: int = Field(default=20, ge=1)
    rag_prompt_version: str = "parse-before-you-prompt-rag-v1"
    rag_max_evidence_tokens: int = Field(default=6000, ge=1)
    rag_max_question_chars: int = Field(default=4000, ge=1)
    chat_temperature: float | None = Field(default=None, ge=0, le=2)
    chat_reasoning_effort: Literal["low"] = "low"
    chat_max_output_tokens: int = Field(default=2000, ge=1)
    chat_structured_output_retry_count: int = Field(default=1, ge=0)

    azure_openai_base_url: str | None = None
    azure_openai_auth_mode: str = "api_key"
    azure_openai_api_key: SecretStr | None = None
    azure_openai_token_scope: str = "https://cognitiveservices.azure.com/.default"
    azure_openai_managed_identity_client_id: str | None = None
    azure_openai_embedding_deployment: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_request_timeout_seconds: int = Field(default=120, ge=1)
    azure_openai_max_retries: int = Field(default=2, ge=0)

    processing_worker_enabled: bool = True
    processing_worker_poll_seconds: float = Field(default=1.0, ge=0.1, le=60)
    processing_worker_concurrency: int = Field(default=1, ge=1)
    processing_stale_after_minutes: int = Field(default=60, ge=1)
    processing_shutdown_timeout_seconds: int = Field(default=30, ge=1, le=300)

    allow_demo_reset: bool = False
    cors_allowed_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    log_level: str = "INFO"

    @field_validator(
        "app_name",
        "app_env",
        "chroma_host",
        "tiktoken_model_hint",
        "tiktoken_fallback_encoding",
        "rag_prompt_version",
        "azure_openai_auth_mode",
        "azure_openai_token_scope",
        mode="before",
    )
    @classmethod
    def strip_required_strings(cls, value: object) -> object:
        """Trim surrounding whitespace from non-secret string settings."""

        return value.strip() if isinstance(value, str) else value

    @field_validator(
        "azure_openai_base_url",
        "azure_openai_embedding_deployment",
        "azure_openai_chat_deployment",
        "azure_openai_managed_identity_client_id",
        mode="before",
    )
    @classmethod
    def strip_optional_strings(cls, value: object) -> object:
        """Normalize blank optional settings to ``None`` without inferring values."""

        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator("azure_openai_api_key", mode="before")
    @classmethod
    def strip_optional_secret(cls, value: object) -> object:
        """Trim an environment-supplied key while retaining SecretStr redaction."""

        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator("azure_openai_auth_mode")
    @classmethod
    def normalize_auth_mode(cls, value: str) -> str:
        """Normalize auth mode while deferring unsupported-mode errors to model calls."""

        return value.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Normalize and validate standard Python logging levels."""

        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"Unsupported log level: {value}")
        return normalized

    @field_validator("processing_worker_concurrency")
    @classmethod
    def validate_worker_concurrency(cls, value: int) -> int:
        """This demonstration intentionally supports exactly one worker."""

        if value != 1:
            raise ValueError("PROCESSING_WORKER_CONCURRENCY must be exactly 1")
        return value

    @property
    def cors_origins(self) -> list[str]:
        """Return normalized exact origins without permitting wildcards."""

        origins = [item.strip().rstrip("/") for item in self.cors_allowed_origins.split(",")]
        normalized = [item for item in origins if item]
        if "*" in normalized:
            raise ValueError("CORS_ALLOWED_ORIGINS must contain exact origins, not '*'")
        return list(dict.fromkeys(normalized))

    @property
    def azure_openai_base_url_validation_error(self) -> str | None:
        """Return a safe validation message without exposing the configured URL."""

        if not self.azure_openai_base_url:
            return None
        try:
            normalize_azure_openai_v1_base_url(self.azure_openai_base_url)
        except ValueError as exc:
            return str(exc)
        return None

    @property
    def azure_openai_embedding_missing_settings(self) -> list[str]:
        """Return settings preventing embedding calls, excluding chat configuration."""

        missing: list[str] = []
        if not self.azure_openai_base_url or self.azure_openai_base_url_validation_error:
            missing.append("AZURE_OPENAI_BASE_URL")
        if not self.azure_openai_embedding_deployment:
            missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if self.azure_openai_auth_mode == "api_key" and self.azure_openai_api_key is None:
            missing.append("AZURE_OPENAI_API_KEY")
        elif self.azure_openai_auth_mode == "entra" and not self.azure_openai_token_scope:
            missing.append("AZURE_OPENAI_TOKEN_SCOPE")
        elif self.azure_openai_auth_mode not in SUPPORTED_AZURE_OPENAI_AUTH_MODES:
            missing.append("AZURE_OPENAI_AUTH_MODE")
        return missing

    @property
    def azure_openai_embedding_ready(self) -> bool:
        """Report embedding readiness independently from chat readiness."""

        return not self.azure_openai_embedding_missing_settings

    @property
    def azure_openai_chat_missing_settings(self) -> list[str]:
        """Return settings preventing chat calls, excluding embedding deployment."""

        missing: list[str] = []
        if not self.azure_openai_base_url or self.azure_openai_base_url_validation_error:
            missing.append("AZURE_OPENAI_BASE_URL")
        if not self.azure_openai_chat_deployment:
            missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT")
        if self.azure_openai_auth_mode == "api_key" and self.azure_openai_api_key is None:
            missing.append("AZURE_OPENAI_API_KEY")
        elif self.azure_openai_auth_mode == "entra" and not self.azure_openai_token_scope:
            missing.append("AZURE_OPENAI_TOKEN_SCOPE")
        elif self.azure_openai_auth_mode not in SUPPORTED_AZURE_OPENAI_AUTH_MODES:
            missing.append("AZURE_OPENAI_AUTH_MODE")
        return missing

    @property
    def azure_openai_chat_ready(self) -> bool:
        """Report chat readiness independently from embedding readiness."""

        return not self.azure_openai_chat_missing_settings

    @property
    def rag_model_ready(self) -> bool:
        """Report global model readiness without implying any run is indexed."""

        return self.azure_openai_embedding_ready and self.azure_openai_chat_ready

    @property
    def azure_openai_missing_settings(self) -> list[str]:
        """Return all model settings missing for the eventual full RAG application."""

        return list(
            dict.fromkeys(
                self.azure_openai_embedding_missing_settings
                + self.azure_openai_chat_missing_settings
            )
        )

    @property
    def azure_openai_configured(self) -> bool:
        """Report whether embedding and future chat settings are both present."""

        return not self.azure_openai_missing_settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
