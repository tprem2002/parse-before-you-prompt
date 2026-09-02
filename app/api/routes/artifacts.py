"""Processing-run artifact lookup endpoints."""

from __future__ import annotations

from uuid import UUID

from pathlib import Path

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.models import Artifact
from app.db.repositories import ArtifactRepository, ProcessingRunRepository
from app.db.session import get_db
from app.schemas.processing import ArtifactResponse
from app.schemas.common import error_responses
from app.core.errors import ApiError
from app.services.api_view_service import artifact_view, resolve_artifact_content


router = APIRouter(tags=["Artifacts"], responses=error_responses(404, 500))
artifact_repository = ArtifactRepository()
run_repository = ProcessingRunRepository()


@router.get(
    "/processing-runs/{run_id}/artifacts",
    response_model=list[ArtifactResponse],
)
def list_processing_artifacts(
    run_id: UUID, session: Session = Depends(get_db)
) -> list[ArtifactResponse]:
    """List filesystem artifacts recorded for one processing run."""

    if run_repository.get(session, run_id) is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "processing_run_not_found",
            "Processing run not found.",
        )
    return [artifact_view(item) for item in artifact_repository.list_for_run(session, run_id)]


@router.get("/artifacts/{artifact_id}/content", response_class=FileResponse)
def stream_artifact_content(
    artifact_id: UUID,
    session: Session = Depends(get_db),
) -> FileResponse:
    """Stream one registered project artifact without accepting filesystem paths."""

    artifact = artifact_repository.get(session, artifact_id)
    if artifact is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "artifact_not_found",
            "Artifact not found.",
        )
    try:
        path, mime_type = resolve_artifact_content(artifact)
    except (FileNotFoundError, OSError, ValueError):
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "artifact_content_unavailable",
            "Artifact content is unavailable.",
        ) from None
    suffix = Path(path.name).suffix.lower()
    return FileResponse(
        path,
        media_type=mime_type,
        filename=f"artifact-{artifact.id}{suffix}",
    )
