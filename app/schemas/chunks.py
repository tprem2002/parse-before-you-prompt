"""Public chunk summary, detail, and provenance contracts."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ProvenanceResponse(BaseModel):
    id: UUID
    doc_item_ref: str
    page_no: int
    bbox_left: float | None
    bbox_top: float | None
    bbox_right: float | None
    bbox_bottom: float | None
    coordinate_origin: str | None
    char_start: int | None
    char_end: int | None
    evidence_role: str


class ChunkSummaryResponse(BaseModel):
    id: UUID
    processing_run_id: UUID
    ordinal: int
    chunk_role: str
    kind: str
    raw_text: str | None = None
    embedding_text: str | None = None
    raw_token_count: int
    contextualized_token_count: int
    page_start: int | None
    page_end: int | None
    content_classification: str
    vector_collection: str | None
    vector_id: str | None
    precise_provenance_available: bool
    overlay_available: bool


class ChunkListResponse(BaseModel):
    items: list[ChunkSummaryResponse]
    total: int
    offset: int
    limit: int


class ChunkDetailResponse(ChunkSummaryResponse):
    document_id: UUID
    token_count: int
    max_token_count: int | None
    section_path: list[Any]
    captions: list[Any]
    doc_item_refs: list[Any]
    table_ref: str | None
    picture_ref: str | None
    is_derived_content: bool
    chunking_fingerprint: str | None
    serializer_metadata: dict[str, Any]
    chunk_metadata: dict[str, Any]
    header_repetition_status: str | None
    overflow: bool
    provenance: list[ProvenanceResponse] = Field(default_factory=list)
    available_evidence_pages: list[int] = Field(default_factory=list)
    evidence_image_url: str | None
    created_at: datetime
