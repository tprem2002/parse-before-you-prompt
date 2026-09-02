"""Azure OpenAI v1 embedding provider with external token-aware batching."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from time import perf_counter
from typing import Any

from openai import OpenAI

from app.core.config import Settings, get_settings, normalize_azure_openai_v1_base_url
from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.providers.azure_openai_client import (
    AzureOpenAIClientHandle,
    build_azure_openai_client,
)
from app.providers.embeddings.base import (
    EmbeddingBatchResult,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    build_embedding_batch_plan,
)
from app.services.baseline_chunker import resolve_tokenizer


PROVIDER_NAME = "azure_openai"
PROVIDER_IMPLEMENTATION_VERSION = "azure-openai-embedding-provider-v1"
INPUT_SERIALIZATION_VERSION = "embedding-text-utf8-v1"
BASELINE_INPUT_REPRESENTATION_VERSION = "baseline-plain-v1"
DOCLING_INPUT_REPRESENTATION_VERSION = "docling-contextualized-v1"

logger = get_logger(__name__)


def canonical_sha256(value: dict[str, object]) -> str:
    """Hash canonical JSON for stable, non-secret identity records."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_base_url_hash(settings: Settings, *, allow_incomplete: bool) -> str | None:
    """Return only a URL hash; never expose the configured endpoint value."""

    if not settings.azure_openai_base_url:
        return None
    try:
        normalized = normalize_azure_openai_v1_base_url(settings.azure_openai_base_url)
    except ValueError:
        if not allow_incomplete:
            raise
        normalized = settings.azure_openai_base_url.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def input_representation_version_for(pipeline_type: str) -> str:
    """Return the truthful text representation identity for an eligible pipeline."""

    versions = {
        "baseline": BASELINE_INPUT_REPRESENTATION_VERSION,
        "docling_standard": DOCLING_INPUT_REPRESENTATION_VERSION,
    }
    try:
        return versions[pipeline_type]
    except KeyError as exc:
        raise EmbeddingConfigurationError(
            f"Pipeline {pipeline_type} has no embedding input representation"
        ) from exc


def provisional_embedding_fingerprint(
    settings: Settings, *, pipeline_type: str
) -> tuple[str, dict[str, object]]:
    """Build the pre-response semantic identity with a redacted endpoint hash."""

    tokenizer = resolve_tokenizer(settings)
    metadata: dict[str, object] = {
        "provider": PROVIDER_NAME,
        "pipeline_type": pipeline_type,
        "normalized_base_url_hash": normalized_base_url_hash(settings, allow_incomplete=True),
        "deployment_name": settings.azure_openai_embedding_deployment,
        "tokenizer_model_hint": settings.tiktoken_model_hint,
        "resolved_tokenizer_encoding": tokenizer.encoding.name,
        "embedding_provider_version": PROVIDER_IMPLEMENTATION_VERSION,
        "input_representation_version": input_representation_version_for(pipeline_type),
        "input_serialization_version": INPUT_SERIALIZATION_VERSION,
    }
    return canonical_sha256(metadata), metadata


def final_embedding_fingerprint(
    settings: Settings,
    *,
    pipeline_type: str,
    service_model: str | None,
    vector_dimension: int,
) -> tuple[str, dict[str, object]]:
    """Build final semantic vector identity after a validated Azure response."""

    tokenizer = resolve_tokenizer(settings)
    metadata: dict[str, object] = {
        "provider": PROVIDER_NAME,
        "pipeline_type": pipeline_type,
        "normalized_base_url_hash": normalized_base_url_hash(settings, allow_incomplete=False),
        "deployment_name": settings.azure_openai_embedding_deployment,
        "service_returned_model": service_model,
        "vector_dimension": vector_dimension,
        "tokenizer_model_hint": settings.tiktoken_model_hint,
        "resolved_tokenizer_encoding": tokenizer.encoding.name,
        "embedding_provider_version": PROVIDER_IMPLEMENTATION_VERSION,
        "input_representation_version": input_representation_version_for(pipeline_type),
        "input_serialization_version": INPUT_SERIALIZATION_VERSION,
    }
    return canonical_sha256(metadata), metadata


class AzureOpenAIEmbeddingProvider:
    """Lazy reusable Azure embedding client using exactly one SDK retry layer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.azure_openai_embedding_ready:
            missing = ", ".join(self._settings.azure_openai_embedding_missing_settings)
            raise EmbeddingConfigurationError(f"Azure OpenAI embeddings are not configured: {missing}")
        try:
            self._base_url = normalize_azure_openai_v1_base_url(
                self._settings.azure_openai_base_url or ""
            )
        except ValueError as exc:
            raise EmbeddingConfigurationError(str(exc)) from exc
        self._client: OpenAI | None = None
        self._client_handle: AzureOpenAIClientHandle | None = None
        self._client_lock = threading.Lock()

    @property
    def deployment_name(self) -> str:
        deployment = self._settings.azure_openai_embedding_deployment
        if deployment is None:
            raise EmbeddingConfigurationError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required")
        return deployment

    @property
    def non_secret_configuration_fingerprint(self) -> str:
        """Identify client construction settings without credentials or endpoint text."""

        return canonical_sha256(
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
            }
        )

    def _get_client(self) -> OpenAI:
        """Initialize the SDK client once, protected against concurrent access."""

        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                self._client_handle = build_azure_openai_client(self._settings)
            except ConfigurationError as exc:
                raise EmbeddingConfigurationError(str(exc)) from exc
            self._client = self._client_handle.client
            return self._client

    def embed_documents(
        self,
        texts: list[str],
        *,
        input_ids: list[str] | None = None,
    ) -> EmbeddingBatchResult:
        """Embed ordered text batches and reject malformed or unsafe responses."""

        tokenizer = resolve_tokenizer(self._settings)
        plan = build_embedding_batch_plan(
            texts,
            input_ids=input_ids,
            stored_token_counts=None,
            encoding=tokenizer.encoding,
            tokenizer_model_hint=tokenizer.model_hint,
            tokenizer_fallback_used=tokenizer.used_fallback,
            max_inputs=self._settings.embedding_batch_max_inputs,
            max_tokens=self._settings.embedding_batch_max_tokens,
            per_input_max_tokens=self._settings.chunk_max_tokens,
        )
        client = self._get_client()
        started = perf_counter()
        vectors: list[tuple[float, ...]] = []
        dimension: int | None = None
        service_models: set[str] = set()
        prompt_tokens = 0
        total_tokens = 0
        usage_available = True
        request_ids: list[str] = []

        for batch in plan.batches:
            batch_started = perf_counter()
            try:
                response = client.embeddings.create(
                    input=texts[batch.start_index : batch.end_index],
                    model=self.deployment_name,
                    encoding_format="float",
                )
                batch_vectors = self._validate_response_data(
                    response.data,
                    expected_count=batch.input_count,
                    expected_dimension=dimension,
                    batch_number=batch.batch_number,
                )
            except EmbeddingProviderError:
                raise
            except Exception as exc:
                raise self._safe_provider_error(exc, batch.batch_number) from None

            if dimension is None:
                dimension = len(batch_vectors[0])
            vectors.extend(batch_vectors)
            if response.model:
                service_models.add(response.model)
            if response.usage is None:
                usage_available = False
            else:
                prompt_tokens += response.usage.prompt_tokens
                total_tokens += response.usage.total_tokens
            request_id = getattr(response, "_request_id", None)
            if isinstance(request_id, str) and request_id:
                request_ids.append(request_id)
            logger.info(
                "Embedding batch=%d inputs=%d tokens=%d duration_ms=%d "
                "status=success request_id=%s retry_count=unknown",
                batch.batch_number,
                batch.input_count,
                batch.aggregate_tokens,
                round((perf_counter() - batch_started) * 1000),
                request_id or "unavailable",
            )

        if dimension is None or len(vectors) != len(texts):
            raise EmbeddingProviderError("Embedding response validation failed: incomplete result")
        if len(service_models) > 1:
            raise EmbeddingProviderError(
                "Embedding response validation failed: service model changed between batches"
            )
        return EmbeddingBatchResult(
            vectors=tuple(vectors),
            vector_dimension=dimension,
            provider_name=PROVIDER_NAME,
            deployment_name=self.deployment_name,
            service_model=next(iter(service_models), None),
            input_count=len(texts),
            aggregate_token_count=plan.aggregate_tokens,
            batch_count=len(plan.batches),
            request_duration_ms=round((perf_counter() - started) * 1000),
            usage_prompt_tokens=prompt_tokens if usage_available else None,
            usage_total_tokens=total_tokens if usage_available else None,
            request_ids=tuple(request_ids),
            retry_count=None,
            tokenizer_model_hint=plan.tokenizer_model_hint,
            tokenizer_encoding=plan.tokenizer_encoding,
            tokenizer_fallback_used=plan.tokenizer_fallback_used,
        )

    def embed_query_result(self, text: str) -> EmbeddingBatchResult:
        """Embed one query while retaining response identity and safe telemetry."""

        return self.embed_documents([text], input_ids=["query"])

    def embed_query(self, text: str) -> list[float]:
        """Embed one text through the same implementation as document vectors."""

        result = self.embed_query_result(text)
        return list(result.vectors[0])

    @staticmethod
    def _validate_response_data(
        data: list[Any],
        *,
        expected_count: int,
        expected_dimension: int | None,
        batch_number: int,
    ) -> list[tuple[float, ...]]:
        if len(data) != expected_count:
            raise EmbeddingProviderError(
                f"Embedding batch {batch_number} returned {len(data)} records; "
                f"expected {expected_count}"
            )
        indexes = [item.index for item in data]
        if len(set(indexes)) != len(indexes) or set(indexes) != set(range(expected_count)):
            raise EmbeddingProviderError(
                f"Embedding batch {batch_number} returned invalid response indexes"
            )
        ordered = sorted(data, key=lambda item: item.index)
        validated: list[tuple[float, ...]] = []
        batch_dimension = expected_dimension
        for item in ordered:
            if not item.embedding:
                raise EmbeddingProviderError(
                    f"Embedding batch {batch_number} returned an empty vector"
                )
            try:
                vector = tuple(float(value) for value in item.embedding)
            except (TypeError, ValueError) as exc:
                raise EmbeddingProviderError(
                    f"Embedding batch {batch_number} returned a non-numeric vector value"
                ) from exc
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingProviderError(
                    f"Embedding batch {batch_number} returned a non-finite vector value"
                )
            if batch_dimension is None:
                batch_dimension = len(vector)
            if len(vector) != batch_dimension:
                raise EmbeddingProviderError(
                    f"Embedding batch {batch_number} returned inconsistent vector dimensions"
                )
            validated.append(vector)
        return validated

    @staticmethod
    def _safe_provider_error(exc: Exception, batch_number: int) -> EmbeddingProviderError:
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
            "Embedding batch=%d status=%s request_id=%s retry_count=sdk-managed",
            batch_number,
            category,
            safe_request_id,
        )
        return EmbeddingProviderError(
            f"Azure embedding batch {batch_number} failed: {category}; "
            f"request_id={safe_request_id}; exception_type={type(exc).__name__}"
        )


_provider_lock = threading.Lock()
_provider_instance: AzureOpenAIEmbeddingProvider | None = None
_provider_fingerprint: str | None = None


def get_azure_openai_embedding_provider(
    settings: Settings | None = None,
) -> AzureOpenAIEmbeddingProvider:
    """Return a shared provider, rebuilding only for non-secret config changes."""

    global _provider_instance, _provider_fingerprint
    configured_settings = settings or get_settings()
    candidate = AzureOpenAIEmbeddingProvider(configured_settings)
    fingerprint = candidate.non_secret_configuration_fingerprint
    with _provider_lock:
        if _provider_instance is None or _provider_fingerprint != fingerprint:
            _provider_instance = candidate
            _provider_fingerprint = fingerprint
        return _provider_instance
