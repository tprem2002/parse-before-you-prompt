"""Shared application and public API exceptions."""

from __future__ import annotations

from typing import Any


class ApplicationError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(ApplicationError):
    """Raised when an operation requires missing configuration."""


class ExternalServiceError(ApplicationError):
    """Raised when a configured external service is unavailable."""


class ApiError(ApplicationError):
    """An expected HTTP failure with a stable, safe response contract."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
