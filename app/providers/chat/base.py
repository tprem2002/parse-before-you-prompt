"""Typed chat provider contract for grounded structured answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.core.errors import ConfigurationError, ExternalServiceError
from app.schemas.rag import AnswerResponse


class ChatConfigurationError(ConfigurationError):
    """Chat execution is not safely configured."""


class ChatProviderError(ExternalServiceError):
    """A chat transport or service request failed safely."""


class ChatOutputValidationError(ChatProviderError):
    """The service returned output that did not match the strict answer schema."""

    def __init__(self, message: str, *, request_duration_ms: int) -> None:
        self.request_duration_ms = request_duration_ms
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class EvidenceForModel:
    """The deliberately minimal evidence shape allowed to leave the application."""

    evidence_id: str
    kind: str
    evidence_class: str
    contextualized_text: str


@dataclass(frozen=True, slots=True)
class ChatAnswerResult:
    answer: AnswerResponse
    provider: str
    deployment: str
    service_model: str | None
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    request_duration_ms: int
    attempt_number: int
    prompt_version: str
    prompt_hash: str
    schema_version: str


class ChatProvider(Protocol):
    def answer(
        self,
        question: str,
        evidence: list[EvidenceForModel],
        *,
        validation_feedback: str | None = None,
        attempt_number: int = 1,
    ) -> ChatAnswerResult:
        ...
