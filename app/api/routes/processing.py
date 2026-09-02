"""Baseline processing request and lifecycle endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.orm import Session

from app.core.enums import IndexMode, PipelineType, ProcessingStatus
from app.core.errors import ApiError
from app.core.config import get_settings
from app.db.models import ProcessingRun
from app.db.repositories import ProcessingRunRepository
from app.db.session import get_db
from app.schemas.processing import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ProcessingRunResponse,
)
from app.schemas.common import error_responses
from app.services.job_service import (
    select_or_create_baseline_run,
    select_or_create_docling_standard_run,
)
from app.services.processing_worker import get_processing_worker
from app.services.api_view_service import processing_run_view


router = APIRouter(
    tags=["Processing"],
    responses=error_responses(404, 409, 501, 503, 500),
)
run_repository = ProcessingRunRepository()


@router.post(
    "/documents/{document_id}/process",
    response_model=ProcessDocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_200_OK: {
            "model": ProcessDocumentResponse,
            "description": "Active or completed work reused.",
        }
    },
    summary="Queue document processing",
)
def process_document(
    document_id: UUID,
    request: ProcessDocumentRequest,
    http_request: Request,
    response: Response,
    session: Session = Depends(get_db),
) -> ProcessDocumentResponse:
    """Queue an implemented processing path or return reusable existing work."""

    if request.pipeline_type is PipelineType.DOCLING_GRANITE_VLM:
        raise ApiError(
            status.HTTP_501_NOT_IMPLEMENTED,
            "granite_docling_deferred",
            "Granite-Docling-258M remains deferred.",
        )
    settings = get_settings()
    from app.db.repositories import DocumentRepository

    if DocumentRepository().get(session, document_id) is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "document_not_found", "Document not found.")
    if request.index_mode is IndexMode.REQUIRED and not settings.azure_openai_embedding_ready:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "embedding_configuration_unavailable",
            "Required indexing is unavailable because embedding configuration is incomplete.",
            details={"missing_settings": settings.azure_openai_embedding_missing_settings},
        )
    try:
        if request.pipeline_type is PipelineType.BASELINE:
            selection = select_or_create_baseline_run(
                session,
                document_id,
                index_mode=request.index_mode,
                force_reprocess=request.force_reprocess,
                retry_failed=request.retry_failed,
                request_id=http_request.state.request_id,
            )
        else:
            selection = select_or_create_docling_standard_run(
                session,
                document_id,
                index_mode=request.index_mode,
                force_reprocess=request.force_reprocess,
                retry_failed=request.retry_failed,
                request_id=http_request.state.request_id,
            )
    except LookupError as exc:
        raise ApiError(status.HTTP_404_NOT_FOUND, "document_not_found", str(exc)) from exc
    except RuntimeError as exc:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "incompatible_active_run",
            str(exc),
        ) from exc
    session.commit()
    session.refresh(selection.run)
    worker_configuration = (selection.run.configuration_json or {}).get("background_processing")
    persisted_mode = (
        worker_configuration.get("index_mode")
        if isinstance(worker_configuration, dict)
        else IndexMode.SKIP.value
    )
    try:
        selected_index_mode = IndexMode(persisted_mode)
    except ValueError:
        selected_index_mode = IndexMode.SKIP

    if selection.should_schedule:
        get_processing_worker().enqueue(
            selection.run.id,
            request_id=http_request.state.request_id,
        )

    if not selection.reused:
        message = (
            f"{request.pipeline_type.value} processing queued with "
            f"index_mode={request.index_mode.value}"
        )
    elif selection.run.status == ProcessingStatus.COMPLETED.value:
        message = (
            f"Reusing completed {request.pipeline_type.value} processing run "
            "with identical source hash and configuration"
        )
    else:
        message = (
            f"Reusing existing {request.pipeline_type.value} processing run in "
            f"{selection.run.status} state with index_mode={selected_index_mode.value}"
        )
    response.status_code = (
        status.HTTP_200_OK if selection.reused else status.HTTP_202_ACCEPTED
    )
    view = processing_run_view(selection.run, session, reused=selection.reused)
    return ProcessDocumentResponse(**view.model_dump(), message=message)


@router.get("/processing-runs/{run_id}", response_model=ProcessingRunResponse)
def get_processing_run(
    run_id: UUID, session: Session = Depends(get_db)
) -> ProcessingRunResponse:
    """Return processing progress and any failure detail."""

    run = run_repository.get(session, run_id)
    if run is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "processing_run_not_found",
            "Processing run not found.",
        )
    return processing_run_view(run, session)
