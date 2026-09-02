"""Service health and safe configuration-status endpoints."""

from __future__ import annotations

from time import perf_counter
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import ProcessingRun
from app.db.session import SessionLocal, check_database_connection
from app.services.chroma_service import get_chroma_service


router = APIRouter()
logger = get_logger(__name__)


class ServiceHealth(BaseModel):
    """Connectivity result for one local dependency."""

    status: Literal["healthy", "unhealthy"]
    latency_ms: float
    detail: str | None = None


class HealthResponse(BaseModel):
    """Aggregate application health response."""

    status: Literal["healthy", "unhealthy"]
    services: dict[str, ServiceHealth]


class AzureConfigurationStatus(BaseModel):
    """Non-secret Azure configuration readiness."""

    configured: bool
    embedding_provider: str
    chat_provider: str
    auth_mode: str
    missing_settings: list[str]
    embedding_ready: bool
    chat_ready: bool
    rag_model_ready: bool
    missing_embedding_settings: list[str]
    missing_chat_settings: list[str]
    base_url_configured: bool
    base_url_valid: bool
    base_url_validation_error: str | None
    api_key_configured: bool
    entra_scope_configured: bool
    embedding_deployment_configured: bool
    chat_deployment_configured: bool


class ConfigurationStatusResponse(BaseModel):
    """Safe runtime configuration summary."""

    status: Literal["configured", "incomplete"]
    app_name: str
    app_env: str
    azure_openai: AzureConfigurationStatus
    database_configured: bool
    chroma_configured: bool
    chroma_reachable: bool
    chroma_collection_count: int | None
    chunk_max_tokens: int
    rag_top_k: int
    indexed_baseline_run_count: int | None
    indexed_docling_standard_run_count: int | None
    processing_worker_enabled: bool
    processing_worker_concurrency: int
    processing_worker_poll_seconds: float
    demo_reset_enabled: bool
    cors_allowed_origins: list[str]


def _database_health() -> ServiceHealth:
    started = perf_counter()
    try:
        check_database_connection()
        return ServiceHealth(
            status="healthy", latency_ms=round((perf_counter() - started) * 1000, 2)
        )
    except Exception as exc:
        logger.warning("PostgreSQL health check failed: %s", type(exc).__name__)
        return ServiceHealth(
            status="unhealthy",
            latency_ms=round((perf_counter() - started) * 1000, 2),
            detail=type(exc).__name__,
        )


def _chroma_health() -> ServiceHealth:
    settings = get_settings()
    started = perf_counter()
    try:
        get_chroma_service(settings).heartbeat()
        return ServiceHealth(
            status="healthy", latency_ms=round((perf_counter() - started) * 1000, 2)
        )
    except Exception as exc:
        logger.warning("Chroma health check failed: %s", type(exc).__name__)
        return ServiceHealth(
            status="unhealthy",
            latency_ms=round((perf_counter() - started) * 1000, 2),
            detail=type(exc).__name__,
        )


def _indexed_run_counts() -> tuple[int | None, int | None]:
    """Count successfully persisted indexes without treating them as model readiness."""

    try:
        with SessionLocal() as session:

            def count(pipeline_type: str) -> int:
                statement = (
                    select(func.count())
                    .select_from(ProcessingRun)
                    .where(
                        ProcessingRun.pipeline_type == pipeline_type,
                        ProcessingRun.configuration_json["embedding_indexing"]["status"].astext
                        == "indexed",
                    )
                )
                return int(session.scalar(statement) or 0)

            return count("baseline"), count("docling_standard")
    except Exception as exc:
        logger.warning("Indexed-run count failed: %s", type(exc).__name__)
        return None, None


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse | JSONResponse:
    """Report PostgreSQL and Chroma connectivity."""

    services = {
        "postgresql": _database_health(),
        "chroma": _chroma_health(),
    }
    overall_status = (
        "healthy" if all(service.status == "healthy" for service in services.values()) else "unhealthy"
    )
    response = HealthResponse(status=overall_status, services=services)
    if overall_status == "unhealthy":
        return JSONResponse(status_code=503, content=response.model_dump())
    return response


@router.get(
    "/configuration/status",
    response_model=ConfigurationStatusResponse,
    tags=["Configuration"],
)
def configuration_status() -> ConfigurationStatusResponse:
    """Report model readiness without returning credentials or configured values."""

    settings = get_settings()
    chroma_status = get_chroma_service(settings).status()
    indexed_baseline, indexed_docling = _indexed_run_counts()
    azure_status = AzureConfigurationStatus(
        configured=settings.azure_openai_configured,
        embedding_provider="azure_openai",
        chat_provider="azure_openai",
        auth_mode=settings.azure_openai_auth_mode,
        missing_settings=settings.azure_openai_missing_settings,
        embedding_ready=settings.azure_openai_embedding_ready,
        chat_ready=settings.azure_openai_chat_ready,
        rag_model_ready=settings.rag_model_ready,
        missing_embedding_settings=settings.azure_openai_embedding_missing_settings,
        missing_chat_settings=settings.azure_openai_chat_missing_settings,
        base_url_configured=bool(settings.azure_openai_base_url),
        base_url_valid=bool(
            settings.azure_openai_base_url
            and settings.azure_openai_base_url_validation_error is None
        ),
        base_url_validation_error=settings.azure_openai_base_url_validation_error,
        api_key_configured=settings.azure_openai_api_key is not None,
        entra_scope_configured=bool(settings.azure_openai_token_scope),
        embedding_deployment_configured=bool(settings.azure_openai_embedding_deployment),
        chat_deployment_configured=bool(settings.azure_openai_chat_deployment),
    )
    return ConfigurationStatusResponse(
        status="configured" if azure_status.configured else "incomplete",
        app_name=settings.app_name,
        app_env=settings.app_env,
        azure_openai=azure_status,
        database_configured=bool(settings.database_url),
        chroma_configured=bool(settings.chroma_host and settings.chroma_port),
        chroma_reachable=chroma_status.reachable,
        chroma_collection_count=chroma_status.collection_count,
        chunk_max_tokens=settings.chunk_max_tokens,
        rag_top_k=settings.rag_top_k,
        indexed_baseline_run_count=indexed_baseline,
        indexed_docling_standard_run_count=indexed_docling,
        processing_worker_enabled=settings.processing_worker_enabled,
        processing_worker_concurrency=settings.processing_worker_concurrency,
        processing_worker_poll_seconds=settings.processing_worker_poll_seconds,
        demo_reset_enabled=settings.allow_demo_reset,
        cors_allowed_origins=settings.cors_origins,
    )
