"""Safe public schemas for document upload and lookup."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentRunSummary(BaseModel):
    id: UUID
    pipeline_type: str
    status: str
    stage: str
    progress_percent: int
    created_at: datetime | None = None
    completed_at: datetime | None = None


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    display_name: str
    mime_type: str
    sha256: str
    file_size_bytes: int
    page_count: int | None
    created_at: datetime
    processing_runs: list[DocumentRunSummary] = Field(default_factory=list)


class UploadResponse(BaseModel):
    document: DocumentResponse
    duplicate: bool
    reused: bool
    message: str
