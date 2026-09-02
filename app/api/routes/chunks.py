"""Paginated chunk inspection and precise evidence-image endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.enums import ArtifactType, ChunkKind, ChunkRole
from app.core.errors import ApiError
from app.db.models import Artifact, Chunk
from app.db.repositories import ChunkRepository, ProcessingRunRepository
from app.db.session import get_db
from app.schemas.chunks import (
    ChunkDetailResponse,
    ChunkListResponse,
    ChunkSummaryResponse,
    ProvenanceResponse,
)
from app.schemas.common import error_responses
from app.services.api_view_service import resolve_artifact_content
from app.services.overlay_service import (
    ChunkNotFoundError,
    OverlayPageRequiredError,
    PreciseProvenanceUnavailableError,
    generate_evidence_overlay,
)


router = APIRouter(tags=["Chunks"], responses=error_responses(400, 404, 422, 500))
chunk_repository = ChunkRepository()
run_repository = ProcessingRunRepository()


def _precise(record) -> bool:  # type: ignore[no-untyped-def]
    return all(
        value is not None
        for value in (
            record.bbox_left,
            record.bbox_top,
            record.bbox_right,
            record.bbox_bottom,
            record.coordinate_origin,
        )
    )


def _overlay_available(session: Session, chunk_id: UUID) -> bool:
    return bool(
        session.scalar(
            select(func.count())
            .select_from(Artifact)
            .where(
                Artifact.artifact_type == ArtifactType.EVIDENCE_OVERLAY.value,
                Artifact.metadata_json["chunk_id"].astext == str(chunk_id),
            )
        )
    )


def _summary(
    chunk: Chunk,
    session: Session,
    *,
    include_text: bool,
) -> ChunkSummaryResponse:
    precise = any(_precise(record) for record in chunk.provenance_records)
    return ChunkSummaryResponse(
        id=chunk.id,
        processing_run_id=chunk.processing_run_id,
        ordinal=chunk.ordinal,
        chunk_role=chunk.chunk_role,
        kind=chunk.kind,
        raw_text=chunk.raw_text if include_text else None,
        embedding_text=chunk.embedding_text if include_text else None,
        raw_token_count=chunk.raw_token_count,
        contextualized_token_count=chunk.contextualized_token_count,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        content_classification=chunk.content_classification,
        vector_collection=chunk.vector_collection,
        vector_id=chunk.vector_id,
        precise_provenance_available=precise,
        overlay_available=_overlay_available(session, chunk.id),
    )


@router.get("/processing-runs/{run_id}/chunks", response_model=ChunkListResponse)
def list_processing_chunks(
    run_id: UUID,
    chunk_role: ChunkRole | None = Query(default=None),
    kind: ChunkKind | None = Query(default=None),
    page: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    include_text: bool = Query(default=False),
    session: Session = Depends(get_db),
) -> ChunkListResponse:
    """Return a bounded, filtered chunk page; text is opt-in."""

    if run_repository.get(session, run_id) is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "processing_run_not_found",
            "Processing run not found.",
        )
    role = chunk_role.value if chunk_role else None
    kind_value = kind.value if kind else None
    chunks = chunk_repository.list_page_for_run(
        session,
        run_id,
        chunk_role=role,
        kind=kind_value,
        page_no=page,
        offset=offset,
        limit=limit,
    )
    total = chunk_repository.count_filtered_for_run(
        session,
        run_id,
        chunk_role=role,
        kind=kind_value,
        page_no=page,
    )
    return ChunkListResponse(
        items=[_summary(chunk, session, include_text=include_text) for chunk in chunks],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/chunks/{chunk_id}", response_model=ChunkDetailResponse)
def get_chunk(chunk_id: UUID, session: Session = Depends(get_db)) -> ChunkDetailResponse:
    """Return full stored and contextualized text with provenance."""

    chunk = chunk_repository.get_with_provenance(session, chunk_id)
    if chunk is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "chunk_not_found", "Chunk not found.")
    base = _summary(chunk, session, include_text=True)
    pages = sorted({record.page_no for record in chunk.provenance_records if _precise(record)})
    return ChunkDetailResponse(
        **base.model_dump(),
        document_id=chunk.document_id,
        token_count=chunk.token_count,
        max_token_count=chunk.max_token_count,
        section_path=chunk.section_path,
        captions=chunk.captions,
        doc_item_refs=chunk.doc_item_refs,
        table_ref=chunk.table_ref,
        picture_ref=chunk.picture_ref,
        is_derived_content=chunk.is_derived_content,
        chunking_fingerprint=chunk.chunking_fingerprint,
        serializer_metadata=chunk.serializer_metadata,
        chunk_metadata=chunk.chunk_metadata,
        header_repetition_status=chunk.header_repetition_status,
        overflow=chunk.overflow,
        provenance=[ProvenanceResponse.model_validate(item, from_attributes=True) for item in chunk.provenance_records],
        available_evidence_pages=pages,
        evidence_image_url=(f"/chunks/{chunk.id}/evidence-image" if pages else None),
        created_at=chunk.created_at,
    )


@router.get("/chunks/{chunk_id}/evidence-image", response_class=FileResponse)
def get_evidence_image(
    chunk_id: UUID,
    page_no: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_db),
) -> FileResponse:
    """Return or deterministically create a provenance-grounded PNG overlay."""

    try:
        overlay = generate_evidence_overlay(session, chunk_id, page_no=page_no)
        session.commit()
        artifact = session.get(Artifact, overlay.artifact_id)
        if artifact is None:
            raise FileNotFoundError
        path, _mime = resolve_artifact_content(artifact)
    except ChunkNotFoundError as exc:
        raise ApiError(status.HTTP_404_NOT_FOUND, "chunk_not_found", str(exc)) from exc
    except OverlayPageRequiredError as exc:
        raise ApiError(
            status.HTTP_400_BAD_REQUEST,
            "evidence_page_required",
            str(exc),
            details={"available_pages": exc.pages},
        ) from exc
    except PreciseProvenanceUnavailableError as exc:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "precise_provenance_unavailable",
            str(exc),
        ) from exc
    except (FileNotFoundError, OSError, ValueError):
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "evidence_image_unavailable",
            "The registered evidence image is unavailable.",
        ) from None
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"chunk-{chunk_id}-page-{overlay.page_no:03d}.png",
        headers={
            "X-Overlay-Artifact-ID": str(overlay.artifact_id),
            "X-Rectangle-Count": str(overlay.rectangle_count),
            "X-Overlay-Reused": str(overlay.reused).lower(),
        },
    )
