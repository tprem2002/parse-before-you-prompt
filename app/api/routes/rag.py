"""RAG query boundary backed by the shared service."""

from fastapi import APIRouter, Request, status

from app.core.config import get_settings
from app.core.errors import ApplicationError, ConfigurationError
from app.core.errors import ApiError
from app.core.logging import get_logger
from app.schemas.rag import RagQueryRequest, RagResponse
from app.schemas.common import error_responses
from app.services.rag_service import GenerationValidationError, answer_question
from app.services.retrieval_service import RunNotIndexedError


router = APIRouter(
    tags=["RAG"],
    responses=error_responses(404, 409, 422, 502, 503, 504, 500),
)
logger = get_logger(__name__)


@router.post("/rag/query", response_model=RagResponse)
def query_rag(request: RagQueryRequest, http_request: Request) -> RagResponse:
    """Delegate the complete grounded query lifecycle to the existing RagService."""

    settings = get_settings()
    if len(request.question) > settings.rag_max_question_chars:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "question_too_long",
            f"Question must not exceed {settings.rag_max_question_chars} characters.",
        )
    if request.top_k is not None and request.top_k > settings.rag_max_top_k:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "top_k_out_of_range",
            f"top_k must not exceed {settings.rag_max_top_k}.",
        )
    logger.info(
        "request_id=%s rag_run_id=%s requested_top_k=%s",
        http_request.state.request_id,
        request.processing_run_id,
        request.top_k or settings.rag_top_k,
    )
    try:
        return answer_question(
            request.processing_run_id,
            request.question,
            top_k=request.top_k,
            settings=settings,
        )
    except LookupError as exc:
        raise ApiError(status.HTTP_404_NOT_FOUND, "processing_run_not_found", str(exc)) from exc
    except RunNotIndexedError as exc:
        raise ApiError(status.HTTP_409_CONFLICT, "run_not_indexed", str(exc)) from exc
    except ConfigurationError as exc:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "model_configuration_unavailable",
            str(exc),
        ) from exc
    except GenerationValidationError as exc:
        raise ApiError(
            status.HTTP_502_BAD_GATEWAY,
            "invalid_upstream_response",
            str(exc),
            details={"query_run_id": str(exc.query_run_id)},
        ) from exc
    except ValueError as exc:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_request", str(exc)) from exc
    except ApplicationError as exc:
        status_code = (
            status.HTTP_504_GATEWAY_TIMEOUT
            if "timeout" in str(exc).lower()
            else status.HTTP_502_BAD_GATEWAY
        )
        code = "upstream_timeout" if status_code == 504 else "upstream_failure"
        raise ApiError(status_code, code, str(exc)) from exc
