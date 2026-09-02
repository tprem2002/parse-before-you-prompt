"""Build redacted API views and safely resolve registered artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models import Artifact, Chunk, Document, ProcessingRun, ProvenanceRecord
from app.schemas.documents import DocumentResponse, DocumentRunSummary
from app.schemas.processing import (
    ArtifactResponse,
    ProcessingRunResponse,
    SafeProcessingError,
)


_MIME_TYPES = {
    "original_pdf": "application/pdf",
    "docling_json": "application/json",
    "conversion_manifest": "application/json",
    "markdown": "text/markdown; charset=utf-8",
    "baseline_text": "text/plain; charset=utf-8",
    "page_image": "image/png",
    "picture_image": "image/png",
    "table_image": "image/png",
    "evidence_overlay": "image/png",
}
_SENSITIVE_KEY_PARTS = (
    "path",
    "endpoint",
    "api_key",
    "authorization",
    "token_scope",
    "access_token",
    "connection",
)


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_metadata(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
        }
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_metadata(item) for item in value]
    return value


def document_view(document: Document, session: Session) -> DocumentResponse:
    runs = list(
        session.scalars(
            select(ProcessingRun)
            .where(ProcessingRun.document_id == document.id)
            .order_by(ProcessingRun.completed_at.desc().nullsfirst())
        )
    )
    summaries: list[DocumentRunSummary] = []
    for run in runs:
        background = (run.configuration_json or {}).get("background_processing")
        queued_at = _parse_datetime(background.get("queued_at")) if isinstance(background, dict) else None
        summaries.append(
            DocumentRunSummary(
                id=run.id,
                pipeline_type=run.pipeline_type,
                status=run.status,
                stage=run.stage,
                progress_percent=run.progress_percent,
                created_at=queued_at,
                completed_at=run.completed_at,
            )
        )
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        display_name=document.display_name,
        mime_type=document.mime_type,
        sha256=document.sha256,
        file_size_bytes=document.file_size_bytes,
        page_count=document.page_count,
        created_at=document.created_at,
        processing_runs=summaries,
    )


def processing_run_view(
    run: ProcessingRun,
    session: Session,
    *,
    reused: bool = False,
) -> ProcessingRunResponse:
    configuration = dict(run.configuration_json or {})
    raw_background = configuration.get("background_processing")
    background = raw_background if isinstance(raw_background, dict) else {}
    raw_indexing = configuration.get("embedding_indexing")
    indexing = raw_indexing if isinstance(raw_indexing, dict) else {}
    status = str(indexing.get("status") or "not_started")
    collection_name = indexing.get("collection_name")
    if not isinstance(collection_name, str):
        collection_name = None
    chunks_by_role = {
        role: int(count)
        for role, count in session.execute(
            select(Chunk.chunk_role, func.count())
            .where(Chunk.processing_run_id == run.id)
            .group_by(Chunk.chunk_role)
        )
    }
    vector_count = int(
        session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(
                Chunk.processing_run_id == run.id,
                Chunk.chunk_role == "vector_index",
                Chunk.vector_id.is_not(None),
            )
        )
        or 0
    )
    failure = background.get("failure")
    recovery = background.get("recovery")
    failure_metadata = failure if isinstance(failure, dict) else recovery
    safe_error = None
    if run.error_message:
        if isinstance(failure_metadata, dict):
            category = str(failure_metadata.get("category") or "processing_failure")
            message = run.error_message
        else:
            category = "processing_failure"
            message = "Processing did not complete successfully. Inspect server logs with the run ID."
        safe_error = SafeProcessingError(category=category, message=message)
    warnings = background.get("warnings")
    safe_warnings = [str(item) for item in warnings] if isinstance(warnings, list) else []
    summary = {
        "pipeline_type": run.pipeline_type,
        "index_mode": background.get("index_mode", "unknown"),
        "parser": configuration.get("parser"),
        "chunk_max_tokens": configuration.get("chunk_max_tokens"),
        "chunk_overlap_tokens": configuration.get("chunk_overlap_tokens"),
        "embedding_provider": indexing.get("provider"),
        "embedding_deployment": indexing.get("deployment_name"),
        "vector_dimension": indexing.get("vector_dimension"),
    }
    return ProcessingRunResponse(
        id=run.id,
        document_id=run.document_id,
        pipeline_type=run.pipeline_type,
        status=run.status,
        stage=run.stage,
        progress_percent=run.progress_percent,
        queued_at=_parse_datetime(background.get("queued_at")),
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_ms=run.duration_ms,
        warnings=safe_warnings,
        error=safe_error,
        indexed=status in {"indexed", "reindexed", "reconciled"} and vector_count > 0,
        indexing_status=status,
        reused=reused,
        configuration_fingerprint=(
            str(
                configuration.get("processing_configuration_fingerprint")
                or configuration.get("configuration_fingerprint")
            )
            if (
                configuration.get("processing_configuration_fingerprint")
                or configuration.get("configuration_fingerprint")
            )
            else None
        ),
        configuration_summary={key: value for key, value in summary.items() if value is not None},
        artifact_count=int(
            session.scalar(
                select(func.count())
                .select_from(Artifact)
                .where(Artifact.processing_run_id == run.id)
            )
            or 0
        ),
        chunks_by_role=chunks_by_role,
        provenance_count=int(
            session.scalar(
                select(func.count())
                .select_from(ProvenanceRecord)
                .where(ProvenanceRecord.processing_run_id == run.id)
            )
            or 0
        ),
        collection_name=collection_name,
        vector_count=vector_count,
        polling_url=f"/processing-runs/{run.id}",
    )


def artifact_view(artifact: Artifact) -> ArtifactResponse:
    path = Path(artifact.storage_path)
    try:
        safe_path, _mime = resolve_artifact_content(artifact)
        size = safe_path.stat().st_size
    except (FileNotFoundError, OSError, ValueError):
        size = 0
    return ArtifactResponse(
        id=artifact.id,
        artifact_type=artifact.artifact_type,
        mime_type=_MIME_TYPES.get(artifact.artifact_type, "application/octet-stream"),
        byte_size=size,
        page_no=artifact.page_no,
        doc_item_ref=artifact.doc_item_ref,
        metadata=_sanitize_metadata(artifact.metadata_json or {}),
        content_url=f"/artifacts/{artifact.id}/content",
    )


def resolve_artifact_content(
    artifact: Artifact,
    settings: Settings | None = None,
) -> tuple[Path, str]:
    """Resolve only a registered file within configured upload/artifact roots."""

    configured = settings or get_settings()
    candidate = Path(artifact.storage_path)
    resolved = candidate.resolve(strict=True)
    roots = (configured.upload_root.resolve(), configured.artifact_root.resolve())
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError("Registered artifact is outside project-managed storage")
    model_cache = configured.hf_home.resolve()
    if resolved == model_cache or model_cache in resolved.parents or resolved.name == ".env":
        raise ValueError("Registered artifact is not eligible for streaming")
    if not resolved.is_file():
        raise FileNotFoundError("Registered artifact file is unavailable")
    return resolved, _MIME_TYPES.get(artifact.artifact_type, "application/octet-stream")
