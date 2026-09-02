"""Background controlled evaluation, polling, and safe exports."""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.errors import ApiError, ConfigurationError
from app.core.logging import get_logger
from app.schemas.common import error_responses
from app.schemas.evaluation import EvaluationRunRequest, EvaluationRunResponse
from app.services.evaluation_service import (
    LIVE_CONFIRMATION,
    TERMINAL_REUSABLE_STATUSES,
    EvaluationConflictError,
    EvaluationNotFoundError,
    GroundTruthValidationError,
    build_evaluation_plan,
    create_or_reuse_evaluation,
    evaluation_csv,
    evaluation_json_payload,
    evaluation_view,
    get_evaluation_executor,
)
from app.services.retrieval_service import RunNotIndexedError


router = APIRouter(tags=["Evaluation"], responses=error_responses(404, 409, 422, 503, 500))
logger = get_logger(__name__)


def _map_preflight_error(exc: Exception) -> ApiError:
    if isinstance(exc, EvaluationNotFoundError):
        return ApiError(status.HTTP_404_NOT_FOUND, "evaluation_resource_not_found", str(exc))
    if isinstance(exc, (EvaluationConflictError, RunNotIndexedError)):
        return ApiError(status.HTTP_409_CONFLICT, "evaluation_conflict", str(exc))
    if isinstance(exc, GroundTruthValidationError):
        return ApiError(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_ground_truth", str(exc))
    if isinstance(exc, ConfigurationError):
        return ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "evaluation_configuration_unavailable", str(exc))
    return ApiError(status.HTTP_500_INTERNAL_SERVER_ERROR, "evaluation_preflight_failed", "Evaluation preflight failed unexpectedly.")


@router.post(
    "/evaluation/run",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        200: {"model": EvaluationRunResponse, "description": "An identical completed evaluation was reused."},
        202: {"model": EvaluationRunResponse, "description": "A new controlled evaluation was queued in the one-worker executor."},
    },
    summary="Run controlled evaluation",
    description=(
        "Runs every repository ground-truth question consecutively against Baseline and Docling standard. "
        "The comparison uses the same text-embedding-3-large deployment, cosine Chroma retrieval, top-k, "
        "GPT-5.1 deployment, prompt, structured schema, citation validator, and retry policy. A new run "
        "makes live Azure query-embedding and chat calls; identical completed evaluations are reused by default."
    ),
)
def run_evaluation(
    request: EvaluationRunRequest,
    response: Response,
    http_request: Request,
) -> EvaluationRunResponse:
    """Validate the fixed experiment, persist one row, and return immediately."""

    if not request.execute:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "evaluation_execute_required",
            "Use scripts/run_evaluation.py --dry-run for a no-call preflight.",
        )
    if request.confirmation != LIVE_CONFIRMATION:
        raise ApiError(
            status.HTTP_400_BAD_REQUEST,
            "invalid_evaluation_confirmation",
            f"Live evaluation requires confirmation={LIVE_CONFIRMATION}.",
        )
    settings = get_settings()
    try:
        plan = build_evaluation_plan(
            document_id=request.document_id,
            baseline_processing_run_id=request.baseline_processing_run_id,
            docling_processing_run_id=request.docling_processing_run_id,
            top_k=request.top_k,
            settings=settings,
        )
        evaluation_id, reused = create_or_reuse_evaluation(plan, force_new=request.force_new)
    except Exception as exc:
        raise _map_preflight_error(exc) from exc
    if reused:
        response.status_code = status.HTTP_200_OK
    else:
        started = get_evaluation_executor(settings).start(evaluation_id, plan)
        if not started:
            raise ApiError(
                status.HTTP_409_CONFLICT,
                "evaluation_executor_busy",
                "Another evaluation is already running in this application process.",
            )
    logger.info(
        "request_id=%s evaluation_id=%s reused=%s total_cases=%s",
        http_request.state.request_id,
        evaluation_id,
        reused,
        plan.total_cases,
    )
    return EvaluationRunResponse.model_validate(
        evaluation_view(evaluation_id, include_results=False, reused=reused)
    )


@router.get(
    "/evaluation/configuration",
    summary="Inspect controlled evaluation configuration",
    description=(
        "Validates the fixed ground truth and selected PostgreSQL index metadata without "
        "querying Chroma, creating an EvaluationRun, or calling Azure."
    ),
)
def get_evaluation_configuration(
    document_id: UUID,
    baseline_processing_run_id: UUID,
    docling_processing_run_id: UUID,
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict[str, object]:
    try:
        plan = build_evaluation_plan(
            document_id=document_id,
            baseline_processing_run_id=baseline_processing_run_id,
            docling_processing_run_id=docling_processing_run_id,
            top_k=top_k,
            settings=get_settings(),
        )
        return plan.as_dict()
    except Exception as exc:
        raise _map_preflight_error(exc) from exc


@router.get(
    "/evaluation/{evaluation_id}",
    response_model=EvaluationRunResponse,
    summary="Poll controlled evaluation",
)
def get_evaluation(
    evaluation_id: UUID,
    include_results: bool = Query(default=False),
) -> EvaluationRunResponse:
    """Return persisted progress and optionally the per-question case rows."""

    try:
        return EvaluationRunResponse.model_validate(
            evaluation_view(evaluation_id, include_results=include_results)
        )
    except EvaluationNotFoundError as exc:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_not_found", "Evaluation not found.") from exc


def _require_exportable(evaluation_id: UUID) -> None:
    try:
        view = evaluation_view(evaluation_id)
    except EvaluationNotFoundError as exc:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_not_found", "Evaluation not found.") from exc
    if view["status"] not in TERMINAL_REUSABLE_STATUSES:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "evaluation_incomplete",
            "Evaluation exports are available only after a valid completed run.",
        )


@router.get("/evaluation/{evaluation_id}/export.csv", summary="Export evaluation CSV")
def export_evaluation_csv(evaluation_id: UUID) -> StreamingResponse:
    """Stream one stable row per pipeline/question case without full evidence text."""

    _require_exportable(evaluation_id)
    content = evaluation_csv(evaluation_id).encode("utf-8-sig")
    return StreamingResponse(
        iter((content,)),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="evaluation-{evaluation_id}.csv"'},
    )


@router.get("/evaluation/{evaluation_id}/export.json", summary="Export evaluation JSON")
def export_evaluation_json(evaluation_id: UUID) -> StreamingResponse:
    """Stream metadata, definitions, aggregates, ranked results, and limitations."""

    _require_exportable(evaluation_id)
    content = json.dumps(
        evaluation_json_payload(evaluation_id), indent=2, ensure_ascii=False
    ).encode("utf-8")
    return StreamingResponse(
        iter((content,)),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="evaluation-{evaluation_id}.json"'},
    )
