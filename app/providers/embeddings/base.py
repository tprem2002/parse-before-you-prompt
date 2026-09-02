"""Typed embedding provider contract and shared deterministic batch planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import tiktoken

from app.core.errors import ConfigurationError, ExternalServiceError


class EmbeddingConfigurationError(ConfigurationError):
    """Embedding execution is not safely configured."""


class EmbeddingProviderError(ExternalServiceError):
    """An embedding request or response failed without exposing input text."""


@dataclass(frozen=True, slots=True)
class TokenCountMismatch:
    """A safe audit record comparing persisted and live token counts."""

    input_id: str
    stored_count: int
    live_count: int


@dataclass(frozen=True, slots=True)
class PlannedEmbeddingBatch:
    """One ordered request batch containing only identifiers and counts."""

    batch_number: int
    start_index: int
    end_index: int
    input_ids: tuple[str, ...]
    token_counts: tuple[int, ...]
    aggregate_tokens: int

    @property
    def input_count(self) -> int:
        return len(self.input_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_number": self.batch_number,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "input_ids": list(self.input_ids),
            "input_count": self.input_count,
            "token_counts": list(self.token_counts),
            "aggregate_tokens": self.aggregate_tokens,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingBatchPlan:
    """Complete deterministic plan built before any remote request."""

    input_ids: tuple[str, ...]
    token_counts: tuple[int, ...]
    batches: tuple[PlannedEmbeddingBatch, ...]
    aggregate_tokens: int
    tokenizer_model_hint: str
    tokenizer_encoding: str
    tokenizer_fallback_used: bool
    token_count_mismatches: tuple[TokenCountMismatch, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    """Validated embeddings and safe request telemetry in original input order."""

    vectors: tuple[tuple[float, ...], ...]
    vector_dimension: int
    provider_name: str
    deployment_name: str
    service_model: str | None
    input_count: int
    aggregate_token_count: int
    batch_count: int
    request_duration_ms: int
    usage_prompt_tokens: int | None
    usage_total_tokens: int | None
    request_ids: tuple[str, ...]
    retry_count: int | None
    tokenizer_model_hint: str
    tokenizer_encoding: str
    tokenizer_fallback_used: bool


class EmbeddingProvider(Protocol):
    """Provider interface shared by document and future query embeddings."""

    def embed_documents(
        self,
        texts: list[str],
        *,
        input_ids: list[str] | None = None,
    ) -> EmbeddingBatchResult:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def build_embedding_batch_plan(
    texts: list[str],
    *,
    input_ids: list[str] | None,
    stored_token_counts: list[int] | None,
    encoding: tiktoken.Encoding,
    tokenizer_model_hint: str,
    tokenizer_fallback_used: bool,
    max_inputs: int,
    max_tokens: int,
    per_input_max_tokens: int,
) -> EmbeddingBatchPlan:
    """Count live tokens and greedily produce stable source-order batches."""

    if not texts:
        raise EmbeddingConfigurationError("At least one embedding input is required")
    ids = input_ids or [f"input-{index}" for index in range(len(texts))]
    if len(ids) != len(texts):
        raise EmbeddingConfigurationError("input_ids must match the embedding input count")
    if len(set(ids)) != len(ids):
        raise EmbeddingConfigurationError("Embedding input identifiers must be unique")
    if stored_token_counts is not None and len(stored_token_counts) != len(texts):
        raise EmbeddingConfigurationError("Stored token counts must match the input count")

    token_counts: list[int] = []
    mismatches: list[TokenCountMismatch] = []
    for index, (input_id, value) in enumerate(zip(ids, texts, strict=True)):
        if not value.strip():
            raise EmbeddingConfigurationError(f"Embedding input {input_id} is empty")
        token_count = len(encoding.encode(value))
        if token_count > per_input_max_tokens:
            raise EmbeddingConfigurationError(
                f"Embedding input {input_id} has {token_count} tokens; "
                f"the configured per-input limit is {per_input_max_tokens}"
            )
        token_counts.append(token_count)
        if stored_token_counts is not None and stored_token_counts[index] != token_count:
            mismatches.append(
                TokenCountMismatch(
                    input_id=input_id,
                    stored_count=stored_token_counts[index],
                    live_count=token_count,
                )
            )

    batches: list[PlannedEmbeddingBatch] = []
    batch_start = 0
    batch_ids: list[str] = []
    batch_counts: list[int] = []
    batch_tokens = 0
    for index, (input_id, token_count) in enumerate(zip(ids, token_counts, strict=True)):
        would_exceed = bool(batch_ids) and (
            len(batch_ids) >= max_inputs or batch_tokens + token_count > max_tokens
        )
        if would_exceed:
            batches.append(
                PlannedEmbeddingBatch(
                    batch_number=len(batches) + 1,
                    start_index=batch_start,
                    end_index=index,
                    input_ids=tuple(batch_ids),
                    token_counts=tuple(batch_counts),
                    aggregate_tokens=batch_tokens,
                )
            )
            batch_start = index
            batch_ids = []
            batch_counts = []
            batch_tokens = 0
        if token_count > max_tokens:
            raise EmbeddingConfigurationError(
                f"Embedding input {input_id} has {token_count} tokens; "
                f"the aggregate batch limit is {max_tokens}"
            )
        batch_ids.append(input_id)
        batch_counts.append(token_count)
        batch_tokens += token_count

    batches.append(
        PlannedEmbeddingBatch(
            batch_number=len(batches) + 1,
            start_index=batch_start,
            end_index=len(texts),
            input_ids=tuple(batch_ids),
            token_counts=tuple(batch_counts),
            aggregate_tokens=batch_tokens,
        )
    )
    return EmbeddingBatchPlan(
        input_ids=tuple(ids),
        token_counts=tuple(token_counts),
        batches=tuple(batches),
        aggregate_tokens=sum(token_counts),
        tokenizer_model_hint=tokenizer_model_hint,
        tokenizer_encoding=encoding.name,
        tokenizer_fallback_used=tokenizer_fallback_used,
        token_count_mismatches=tuple(mismatches),
    )
