"""Azure OpenAI v1 Responses API provider with strict Pydantic output."""

from __future__ import annotations

import hashlib
import json
import threading
from time import perf_counter
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from app.core.config import Settings, get_settings, normalize_azure_openai_v1_base_url
from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.prompts import load_rag_prompt
from app.providers.azure_openai_client import (
    AzureOpenAIClientHandle,
    build_azure_openai_client,
)
from app.providers.chat.base import (
    ChatAnswerResult,
    ChatConfigurationError,
    ChatOutputValidationError,
    ChatProviderError,
    EvidenceForModel,
)
from app.schemas.rag import AnswerResponse


PROVIDER_NAME = "azure_openai"
PROVIDER_IMPLEMENTATION_VERSION = "azure-openai-responses-parse-v1"
logger = get_logger(__name__)


def _canonical_hash(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AzureOpenAIChatProvider:
    """Lazy reusable GPT-4.1 deployment client using Responses API parsing."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.azure_openai_chat_ready:
            missing = ", ".join(self._settings.azure_openai_chat_missing_settings)
            raise ChatConfigurationError(f"Azure OpenAI chat is not configured: {missing}")
        try:
            self._base_url = normalize_azure_openai_v1_base_url(
                self._settings.azure_openai_base_url or ""
            )
        except ValueError as exc:
            raise ChatConfigurationError(str(exc)) from exc
        self._prompt = load_rag_prompt(self._settings.rag_prompt_version)
        self._client_handle: AzureOpenAIClientHandle | None = None
        self._client: OpenAI | None = None
        self._client_lock = threading.Lock()

    @property
    def deployment_name(self) -> str:
        deployment = self._settings.azure_openai_chat_deployment
        if deployment is None:
            raise ChatConfigurationError("AZURE_OPENAI_CHAT_DEPLOYMENT is required")
        return deployment

    @property
    def non_secret_configuration_fingerprint(self) -> str:
        return _canonical_hash(
            {
                "base_url_hash": hashlib.sha256(self._base_url.encode("utf-8")).hexdigest(),
                "auth_mode": self._settings.azure_openai_auth_mode,
                "deployment_name": self.deployment_name,
                "token_scope_hash": hashlib.sha256(
                    self._settings.azure_openai_token_scope.encode("utf-8")
                ).hexdigest(),
                "managed_identity_client_id_hash": (
                    hashlib.sha256(
                        self._settings.azure_openai_managed_identity_client_id.encode("utf-8")
                    ).hexdigest()
                    if self._settings.azure_openai_managed_identity_client_id
                    else None
                ),
                "timeout_seconds": self._settings.azure_openai_request_timeout_seconds,
                "max_retries": self._settings.azure_openai_max_retries,
                "temperature": self._settings.chat_temperature,
                "reasoning_effort": self._settings.chat_reasoning_effort,
                "max_output_tokens": self._settings.chat_max_output_tokens,
                "prompt_hash": self._prompt.sha256,
                "provider_version": PROVIDER_IMPLEMENTATION_VERSION,
            }
        )

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                try:
                    self._client_handle = build_azure_openai_client(self._settings)
                except ConfigurationError as exc:
                    raise ChatConfigurationError(str(exc)) from exc
                self._client = self._client_handle.client
            return self._client

    @staticmethod
    def _model_input(
        question: str,
        evidence: list[EvidenceForModel],
        validation_feedback: str | None,
    ) -> str:
        if not question.strip():
            raise ChatConfigurationError("Question must not be empty")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ChatConfigurationError("Evidence IDs must be unique")
        blocks = [
            "\n".join(
                (
                    f"[{item.evidence_id}]",
                    f"Kind: {item.kind}",
                    f"Evidence class: {item.evidence_class}",
                    "Contextualized evidence text:",
                    item.contextualized_text,
                )
            )
            for item in evidence
        ]
        parts = ["Question:", question.strip(), "", "Ranked evidence:", "\n\n".join(blocks)]
        if validation_feedback:
            parts.extend(("", "Validation feedback:", validation_feedback.strip()))
        return "\n".join(parts)

    @staticmethod
    def _parsed_answer(response: object) -> AnswerResponse:
        parsed: list[AnswerResponse] = []
        for output in getattr(response, "output", []):
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", []):
                value = getattr(content, "parsed", None)
                if isinstance(value, AnswerResponse):
                    parsed.append(value)
        if len(parsed) != 1:
            raise ValueError("Response did not contain exactly one parsed answer")
        return parsed[0]

    def answer(
        self,
        question: str,
        evidence: list[EvidenceForModel],
        *,
        validation_feedback: str | None = None,
        attempt_number: int = 1,
    ) -> ChatAnswerResult:
        """Request one non-streaming structured answer; SDK transport retries remain internal."""

        request_input = self._model_input(question, evidence, validation_feedback)
        started = perf_counter()
        try:
            request_parameters: dict[str, Any] = {
                "model": self.deployment_name,
                "instructions": self._prompt.text,
                "input": request_input,
                "text_format": AnswerResponse,
                "reasoning": {"effort": self._settings.chat_reasoning_effort},
                "max_output_tokens": self._settings.chat_max_output_tokens,
                "tools": [],
                "store": False,
            }
            if self._settings.chat_temperature is not None:
                request_parameters["temperature"] = self._settings.chat_temperature
            response = self._get_client().responses.parse(
                **request_parameters,
            )
            answer = self._parsed_answer(response)
        except (ValidationError, ValueError) as exc:
            duration_ms = round((perf_counter() - started) * 1000)
            logger.warning(
                "Chat attempt=%d status=structured_output_invalid exception_type=%s",
                attempt_number,
                type(exc).__name__,
            )
            raise ChatOutputValidationError(
                "Azure chat returned invalid structured output",
                request_duration_ms=duration_ms,
            ) from None
        except ChatProviderError:
            raise
        except Exception as exc:
            duration_ms = round((perf_counter() - started) * 1000)
            raise self._safe_provider_error(exc, attempt_number, duration_ms) from None

        duration_ms = round((perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        response_id = getattr(response, "id", None)
        logger.info(
            "Chat attempt=%d status=success duration_ms=%d response_id=%s",
            attempt_number,
            duration_ms,
            response_id or "unavailable",
        )
        return ChatAnswerResult(
            answer=answer,
            provider=PROVIDER_NAME,
            deployment=self.deployment_name,
            service_model=getattr(response, "model", None),
            response_id=response_id if isinstance(response_id, str) else None,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            request_duration_ms=duration_ms,
            attempt_number=attempt_number,
            prompt_version=self._prompt.version,
            prompt_hash=self._prompt.sha256,
            schema_version=self._prompt.schema_version,
        )

    @staticmethod
    def _safe_provider_error(
        exc: Exception,
        attempt_number: int,
        duration_ms: int,
    ) -> ChatProviderError:
        status_code = getattr(exc, "status_code", None)
        request_id = getattr(exc, "request_id", None)
        if status_code in {401, 403}:
            category = "authentication_or_permission"
        elif status_code == 404:
            category = "deployment_or_endpoint_not_found"
        elif status_code == 429:
            category = "rate_limited_after_sdk_retries"
        elif isinstance(status_code, int) and status_code >= 500:
            category = "transient_service_failure_after_sdk_retries"
        elif isinstance(status_code, int) and 400 <= status_code < 500:
            category = "invalid_request"
        else:
            category = "transport_or_service_failure"
        safe_request_id = request_id if isinstance(request_id, str) and request_id else "unavailable"
        logger.warning(
            "Chat attempt=%d status=%s duration_ms=%d request_id=%s retry_count=sdk-managed",
            attempt_number,
            category,
            duration_ms,
            safe_request_id,
        )
        return ChatProviderError(
            f"Azure chat failed: {category}; request_id={safe_request_id}; "
            f"exception_type={type(exc).__name__}"
        )


_provider_lock = threading.Lock()
_provider_instance: AzureOpenAIChatProvider | None = None
_provider_fingerprint: str | None = None


def get_azure_openai_chat_provider(
    settings: Settings | None = None,
) -> AzureOpenAIChatProvider:
    """Return a shared provider, rebuilding only for non-secret configuration changes."""

    global _provider_instance, _provider_fingerprint
    configured_settings = settings or get_settings()
    candidate = AzureOpenAIChatProvider(configured_settings)
    fingerprint = candidate.non_secret_configuration_fingerprint
    with _provider_lock:
        if _provider_instance is None or _provider_fingerprint != fingerprint:
            _provider_instance = candidate
            _provider_fingerprint = fingerprint
        return _provider_instance
