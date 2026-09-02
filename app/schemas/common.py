"""Shared public API error contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Safe, stable failure information returned by every API route."""

    code: str
    message: str
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "run_not_indexed",
                        "message": "The processing run has not been indexed.",
                        "request_id": "demo-request-123",
                        "details": {},
                    }
                }
            ]
        }
    )

    error: ErrorDetail


def error_responses(*status_codes: int) -> dict[int, dict[str, object]]:
    """Build reusable OpenAPI response declarations around the shared model."""

    return {
        status_code: {
            "model": ErrorResponse,
            "description": "Safe error response with a request correlation ID.",
        }
        for status_code in status_codes
    }
