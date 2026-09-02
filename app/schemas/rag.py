"""Strict structured generation and RAG response schemas."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonEmptyCitationId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrictAnswerSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClaimResponse(StrictAnswerSchema):
    """One factual claim and the retrieved evidence IDs asserted to support it."""

    text: NonEmptyText
    citation_ids: list[NonEmptyCitationId]


class AnswerResponse(StrictAnswerSchema):
    """The only structured answer shape accepted from the chat deployment."""

    answer: NonEmptyText
    claims: list[ClaimResponse]
    insufficient_evidence: bool


class RagQueryRequest(StrictSchema):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "processing_run_id": "00000000-0000-4000-8000-000000000003",
                    "question": "What is the maximum recovery window?",
                    "top_k": 5,
                }
            ]
        },
    )

    processing_run_id: UUID
    question: NonEmptyText
    top_k: int | None = Field(default=None, ge=1)


class EvidenceImageEndpoint(StrictSchema):
    path: str
    available_pages: list[int]
    cached_overlay_available: bool


class RetrievalEvidenceResponse(StrictSchema):
    evidence_id: str
    rank: int
    chunk_id: UUID
    kind: str
    distance: float
    page_start: int | None
    page_end: int | None
    section_path: list[Any]
    source_captions: list[str]
    source_classification: str
    precise_provenance_available: bool
    provenance_region_count: int
    available_provenance_pages: list[int]
    table_ref: str | None
    picture_ref: str | None
    evidence_image: EvidenceImageEndpoint | None
    contextualized_token_count: int


class ResolvedCitationResponse(RetrievalEvidenceResponse):
    doc_item_refs: list[str]


class TokenUsageResponse(StrictSchema):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class RagResponse(StrictSchema):
    query_run_id: UUID
    processing_run_id: UUID
    pipeline_type: str
    question: str
    requested_top_k: int
    actual_hit_count: int
    retrieval_hits: list[RetrievalEvidenceResponse]
    answer: str
    claims: list[ClaimResponse]
    citation_ids: list[str]
    resolved_citations: list[ResolvedCitationResponse]
    insufficient_evidence: bool
    citation_validation_status: str
    validation_attempt_count: int
    retrieval_duration_ms: int
    generation_duration_ms: int
    total_duration_ms: int
    prompt_version: str
    prompt_hash: str
    schema_version: str
    provider: str | None
    deployment: str | None
    service_model: str | None
    response_id: str | None
    usage: TokenUsageResponse
    total_evidence_tokens: int
