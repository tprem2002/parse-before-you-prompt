"""Disabled-by-default administrative endpoint for the local demo."""

from fastapi import APIRouter, status

from app.core.config import get_settings
from app.core.errors import ApiError
from app.schemas.admin import ResetDemoRequest, ResetDemoResponse
from app.schemas.common import error_responses
from app.services.processing_worker import get_processing_worker
from app.services.reset_service import reset_demo


router = APIRouter(
    prefix="/admin",
    tags=["Demo Administration"],
    responses=error_responses(403, 409, 422, 500),
)


@router.post("/reset-demo", response_model=ResetDemoResponse)
def reset_demo_endpoint(request: ResetDemoRequest) -> ResetDemoResponse:
    """Plan or execute cleanup of only registered project-owned resources."""

    settings = get_settings()
    worker = get_processing_worker(settings).status()
    if worker.active_run_id is not None or worker.pending_count:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "processing_active",
            "Demo reset is unavailable while processing work is active or queued.",
        )
    if not request.dry_run and not settings.allow_demo_reset:
        raise ApiError(
            status.HTTP_403_FORBIDDEN,
            "demo_reset_disabled",
            "Demo reset execution is disabled by configuration.",
        )
    try:
        return reset_demo(dry_run=request.dry_run, settings=settings)
    except ValueError as exc:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "unsafe_reset_scope",
            str(exc),
        ) from exc
