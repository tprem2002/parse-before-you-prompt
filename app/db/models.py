"""Complete relational model for document processing and RAG audit data."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import (
    ArtifactType,
    ChunkKind,
    ChunkRole,
    ContentClassification,
    HeaderRepetitionStatus,
    PipelineType,
    ProcessingStatus,
    ProvenanceRole,
)
from app.db.base import Base


PIPELINE_VALUES = ", ".join(f"'{value.value}'" for value in PipelineType)
PROCESSING_STATUS_VALUES = ", ".join(f"'{value.value}'" for value in ProcessingStatus)
CHUNK_ROLE_VALUES = ", ".join(f"'{value.value}'" for value in ChunkRole)
CHUNK_KIND_VALUES = ", ".join(f"'{value.value}'" for value in ChunkKind)
ARTIFACT_TYPE_VALUES = ", ".join(f"'{value.value}'" for value in ArtifactType)
CONTENT_CLASSIFICATION_VALUES = ", ".join(
    f"'{value.value}'" for value in ContentClassification
)
HEADER_REPETITION_VALUES = ", ".join(
    f"'{value.value}'" for value in HeaderRepetitionStatus
)
PROVENANCE_ROLE_VALUES = ", ".join(f"'{value.value}'" for value in ProvenanceRole)


class Document(Base):
    """An uploaded source PDF and its stable filesystem identity."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(127), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    processing_runs: Mapped[list[ProcessingRun]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    query_runs: Mapped[list[QueryRun]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    evaluation_runs: Mapped[list[EvaluationRun]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class ProcessingRun(Base):
    """A single document conversion attempt and its audit state."""

    __tablename__ = "processing_runs"
    __table_args__ = (
        CheckConstraint(f"pipeline_type IN ({PIPELINE_VALUES})", name="pipeline_type"),
        CheckConstraint(f"status IN ({PROCESSING_STATUS_VALUES})", name="status"),
        CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100", name="progress_percent"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ProcessingStatus.QUEUED.value
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb")
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    error_message: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="processing_runs")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="processing_run", cascade="all, delete-orphan", passive_deletes=True
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="processing_run", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    """A structure-inspection or vector-index chunk with source metadata."""

    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint(f"chunk_role IN ({CHUNK_ROLE_VALUES})", name="chunk_role"),
        CheckConstraint(f"kind IN ({CHUNK_KIND_VALUES})", name="kind"),
        CheckConstraint("ordinal >= 0", name="ordinal"),
        CheckConstraint("token_count >= 0", name="token_count"),
        CheckConstraint("raw_token_count >= 0", name="raw_token_count"),
        CheckConstraint("contextualized_token_count >= 0", name="contextualized_token_count"),
        CheckConstraint(
            "max_token_count IS NULL OR max_token_count >= 1", name="max_token_count"
        ),
        CheckConstraint(
            f"content_classification IN ({CONTENT_CLASSIFICATION_VALUES})",
            name="content_classification",
        ),
        CheckConstraint(
            "header_repetition_status IS NULL OR "
            f"header_repetition_status IN ({HEADER_REPETITION_VALUES})",
            name="header_repetition_status",
        ),
        UniqueConstraint("processing_run_id", "chunk_role", "ordinal", name="run_role_ordinal"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    processing_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processing_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_role: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    contextualized_token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_token_count: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=sql_text("'[]'::jsonb")
    )
    captions: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=sql_text("'[]'::jsonb")
    )
    doc_item_refs: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=sql_text("'[]'::jsonb")
    )
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    table_ref: Mapped[str | None] = mapped_column(Text)
    picture_ref: Mapped[str | None] = mapped_column(Text)
    is_derived_content: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sql_text("false")
    )
    content_classification: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=ContentClassification.SOURCE.value,
        server_default=sql_text("'source'"),
    )
    chunking_fingerprint: Mapped[str | None] = mapped_column(String(64), index=True)
    serializer_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb")
    )
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb")
    )
    header_repetition_status: Mapped[str | None] = mapped_column(String(16))
    overflow: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sql_text("false")
    )
    vector_collection: Mapped[str | None] = mapped_column(String(255))
    vector_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    processing_run: Mapped[ProcessingRun] = relationship(back_populates="chunks")
    provenance_records: Mapped[list[ProvenanceRecord]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", passive_deletes=True
    )
    retrieval_hits: Mapped[list[RetrievalHit]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", passive_deletes=True
    )


class ProvenanceRecord(Base):
    """A source region associated with a chunk."""

    __tablename__ = "provenance_records"
    __table_args__ = (
        CheckConstraint("page_no >= 1", name="page_no"),
        CheckConstraint("char_start IS NULL OR char_start >= 0", name="char_start"),
        CheckConstraint("char_end IS NULL OR char_end >= 0", name="char_end"),
        CheckConstraint(f"evidence_role IN ({PROVENANCE_ROLE_VALUES})", name="evidence_role"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    processing_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processing_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doc_item_ref: Mapped[str] = mapped_column(Text, nullable=False)
    page_no: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_left: Mapped[float | None] = mapped_column(Float)
    bbox_top: Mapped[float | None] = mapped_column(Float)
    bbox_right: Mapped[float | None] = mapped_column(Float)
    bbox_bottom: Mapped[float | None] = mapped_column(Float)
    coordinate_origin: Mapped[str | None] = mapped_column(String(32))
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    evidence_role: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ProvenanceRole.DIRECT_SOURCE_TEXT.value,
        server_default=sql_text("'direct_source_text'"),
    )

    chunk: Mapped[Chunk] = relationship(back_populates="provenance_records")


class Artifact(Base):
    """A filesystem artifact produced or used by a processing run."""

    __tablename__ = "artifacts"
    __table_args__ = (
        CheckConstraint(f"artifact_type IN ({ARTIFACT_TYPE_VALUES})", name="artifact_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    processing_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processing_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    page_no: Mapped[int | None] = mapped_column(Integer)
    doc_item_ref: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    processing_run: Mapped[ProcessingRun] = relationship(back_populates="artifacts")


class QueryRun(Base):
    """A persisted retrieval and answer-generation request."""

    __tablename__ = "query_runs"
    __table_args__ = (
        CheckConstraint(f"pipeline_type IN ({PIPELINE_VALUES})", name="pipeline_type"),
        CheckConstraint("top_k >= 1", name="top_k"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_type: Mapped[str] = mapped_column(String(32), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text)
    answer_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    insufficient_evidence: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=sql_text("false")
    )
    retrieval_duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    generation_duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="query_runs")
    retrieval_hits: Mapped[list[RetrievalHit]] = relationship(
        back_populates="query_run", cascade="all, delete-orphan", passive_deletes=True
    )


class RetrievalHit(Base):
    """A ranked chunk returned for a query run."""

    __tablename__ = "retrieval_hits"
    __table_args__ = (
        CheckConstraint("rank >= 1", name="rank"),
        UniqueConstraint("query_run_id", "rank", name="query_rank"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    distance: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    query_run: Mapped[QueryRun] = relationship(back_populates="retrieval_hits")
    chunk: Mapped[Chunk] = relationship(back_populates="retrieval_hits")


class EvaluationRun(Base):
    """Persisted evaluation results for a document."""

    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    results_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="evaluation_runs")
