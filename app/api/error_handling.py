"""Request correlation, safe logging, and shared FastAPI error rendering."""

from __future__ import annotations

import re
import uuid
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import ApiError
from app.core.logging import get_logger


logger = get_logger(__name__)
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def request_id_for(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else str(uuid.uuid4())


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    request_id = request_id_for(request)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "details": details or {},
            }
        },
        headers={"X-Request-ID": request_id},
    )


def install_error_handling(app: FastAPI) -> None:
    """Install correlation middleware and sanitized exception handlers."""

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        supplied = request.headers.get("X-Request-ID", "").strip()
        request_id = supplied if _SAFE_REQUEST_ID.fullmatch(supplied) else str(uuid.uuid4())
        request.state.request_id = request_id
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_id=%s method=%s path=%s status=unhandled",
                request_id,
                request.method,
                request.url.path,
            )
            raise
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%d",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            round((perf_counter() - started) * 1000),
        )
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            message = str(exc.detail.get("message", "The request could not be completed."))
            details = {key: value for key, value in exc.detail.items() if key != "message"}
        else:
            message = str(exc.detail)
            details = {}
        code = {
            400: "invalid_operation",
            404: "resource_not_found",
            409: "conflict",
            413: "upload_too_large",
            415: "unsupported_media_type",
            422: "invalid_request",
            501: "feature_deferred",
            502: "upstream_response_failure",
            503: "dependency_unavailable",
            504: "upstream_timeout",
        }.get(exc.status_code, "request_failed")
        return error_response(
            request,
            status_code=exc.status_code,
            code=code,
            message=message,
            details=details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {
                "location": [str(item) for item in error.get("loc", ())],
                "type": error.get("type", "validation_error"),
                "message": error.get("msg", "Invalid value"),
            }
            for error in exc.errors()
        ]
        return error_response(
            request,
            status_code=422,
            code="validation_error",
            message="The request did not satisfy the API contract.",
            details={"errors": errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "request_id=%s status=internal_error exception_type=%s",
            request_id_for(request),
            type(exc).__name__,
        )
        return error_response(
            request,
            status_code=500,
            code="internal_error",
            message="The request could not be completed because of an internal error.",
        )
