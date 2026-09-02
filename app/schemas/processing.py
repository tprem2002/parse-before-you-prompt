"""Public processing-run and artifact contracts."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import IndexMode, PipelineType, ProcessingStatus


class ProcessDocumentRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "pipeline_type": "docling_standard",
                    "index_mode": "auto",
                    "force_reprocess": False,
                    "retry_failed": False,
                }
            ]
        },
    )

    pipeline_type: PipelineType
    index_mode: IndexMode = IndexMode.AUTO
    force_reprocess: bool = False
    retry_failed: bool = False


class SafeProcessingError(BaseModel):
    category: str
    message: str


class ProcessingRunResponse(BaseModel):
    id: UUID
    document_id: UUID
    pipeline_type: PipelineType
    status: ProcessingStatus
    stage: str
    progress_percent: int
    queued_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: int | None
    warnings: list[str] = Field(default_factory=list)
    error: SafeProcessingError | None
    indexed: bool
    indexing_status: str
    reused: bool = False
    configuration_fingerprint: str | None
    configuration_summary: dict[str, Any]
    artifact_count: int
    chunks_by_role: dict[str, int]
    provenance_count: int
    collection_name: str | None
    vector_count: int
    polling_url: str


class ProcessDocumentResponse(ProcessingRunResponse):
    message: str


class ArtifactResponse(BaseModel):
    id: UUID
    artifact_type: str
    mime_type: str
    byte_size: int
    page_no: int | None
    doc_item_ref: str | None
    metadata: dict[str, Any]
    content_url: str
