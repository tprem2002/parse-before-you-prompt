"""Small synchronous repository helpers for persistence models."""

from __future__ import annotations

import uuid
from typing import Generic, TypeVar

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.base import Base
from app.db.models import (
    Artifact,
    Chunk,
    Document,
    EvaluationRun,
    ProcessingRun,
    ProvenanceRecord,
    QueryRun,
    RetrievalHit,
)


ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Common persistence operations without owning transaction boundaries."""

    def __init__(self, model: type[ModelT]) -> None:
        self.model = model

    def add(self, session: Session, entity: ModelT) -> ModelT:
        """Stage an entity and flush it so generated values are available."""

        session.add(entity)
        session.flush()
        return entity

    def get(self, session: Session, identifier: uuid.UUID) -> ModelT | None:
        """Return an entity by primary key."""

        return session.get(self.model, identifier)

    def list(self, session: Session, *, limit: int = 100, offset: int = 0) -> list[ModelT]:
        """Return a bounded page of entities."""

        statement = select(self.model).offset(offset).limit(limit)
        return list(session.scalars(statement))

    def delete(self, session: Session, entity: ModelT) -> None:
        """Stage an entity for deletion."""

        session.delete(entity)


class DocumentRepository(BaseRepository[Document]):
    """Persistence helpers for uploaded documents."""

    def __init__(self) -> None:
        super().__init__(Document)

    def get_by_sha256(self, session: Session, sha256: str) -> Document | None:
        """Return the document with a matching content hash."""

        return session.scalar(select(Document).where(Document.sha256 == sha256))

    def get_for_update(self, session: Session, identifier: uuid.UUID) -> Document | None:
        """Lock one document while a processing run is selected or created."""

        statement = select(Document).where(Document.id == identifier).with_for_update()
        return session.scalar(statement)

    def list(self, session: Session, *, limit: int = 100, offset: int = 0) -> list[Document]:
        """Return documents newest first."""

        statement = select(Document).order_by(Document.created_at.desc()).offset(offset).limit(limit)
        return list(session.scalars(statement))


class ProcessingRunRepository(BaseRepository[ProcessingRun]):
    """Persistence helpers for processing runs."""

    def __init__(self) -> None:
        super().__init__(ProcessingRun)

    def list_for_document(
        self, session: Session, document_id: uuid.UUID
    ) -> list[ProcessingRun]:
        """Return processing runs for one document."""

        statement = select(ProcessingRun).where(ProcessingRun.document_id == document_id)
        return list(session.scalars(statement))

    def get_active(
        self,
        session: Session,
        document_id: uuid.UUID,
        pipeline_type: str,
        configuration: dict[str, object],
    ) -> ProcessingRun | None:
        """Return queued/running work for the exact pipeline configuration."""

        statement = (
            select(ProcessingRun)
            .where(
                ProcessingRun.document_id == document_id,
                ProcessingRun.pipeline_type == pipeline_type,
                ProcessingRun.status.in_(("queued", "running")),
                ProcessingRun.configuration_json == configuration,
            )
            .order_by(ProcessingRun.started_at.desc().nullsfirst())
            .limit(1)
        )
        return session.scalar(statement)

    def get_completed(
        self,
        session: Session,
        document_id: uuid.UUID,
        pipeline_type: str,
        configuration: dict[str, object],
    ) -> ProcessingRun | None:
        """Return the newest successful run for an exact pipeline configuration."""

        statement = (
            select(ProcessingRun)
            .where(
                ProcessingRun.document_id == document_id,
                ProcessingRun.pipeline_type == pipeline_type,
                ProcessingRun.status == "completed",
                ProcessingRun.configuration_json.op("-")("embedding_indexing") == configuration,
            )
            .order_by(ProcessingRun.completed_at.desc())
            .limit(1)
        )
        return session.scalar(statement)

    def get_active_baseline(
        self, session: Session, document_id: uuid.UUID
    ) -> ProcessingRun | None:
        """Return an existing queued or running baseline operation."""

        statement = (
            select(ProcessingRun)
            .where(
                ProcessingRun.document_id == document_id,
                ProcessingRun.pipeline_type == "baseline",
                ProcessingRun.status.in_(("queued", "running")),
            )
            .order_by(ProcessingRun.started_at.desc().nullsfirst())
            .limit(1)
        )
        return session.scalar(statement)

    def get_completed_baseline(
        self,
        session: Session,
        document_id: uuid.UUID,
        configuration: dict[str, object],
    ) -> ProcessingRun | None:
        """Return the latest completed run with the exact baseline configuration."""

        statement = (
            select(ProcessingRun)
            .where(
                ProcessingRun.document_id == document_id,
                ProcessingRun.pipeline_type == "baseline",
                ProcessingRun.status == "completed",
                ProcessingRun.configuration_json.op("-")("embedding_indexing") == configuration,
            )
            .order_by(ProcessingRun.completed_at.desc())
            .limit(1)
        )
        return session.scalar(statement)


class ChunkRepository(BaseRepository[Chunk]):
    """Persistence helpers for chunks."""

    def __init__(self) -> None:
        super().__init__(Chunk)

    def list_for_run(
        self,
        session: Session,
        processing_run_id: uuid.UUID,
        *,
        chunk_role: str | None = None,
        kind: str | None = None,
        page_no: int | None = None,
    ) -> list[Chunk]:
        """Return optionally filtered chunks with provenance eagerly loaded."""

        statement = (
            select(Chunk)
            .where(Chunk.processing_run_id == processing_run_id)
            .options(selectinload(Chunk.provenance_records))
            .order_by(Chunk.chunk_role, Chunk.ordinal)
        )
        if chunk_role is not None:
            statement = statement.where(Chunk.chunk_role == chunk_role)
        if kind is not None:
            statement = statement.where(Chunk.kind == kind)
        if page_no is not None:
            statement = statement.where(
                Chunk.page_start.is_not(None),
                Chunk.page_end.is_not(None),
                Chunk.page_start <= page_no,
                Chunk.page_end >= page_no,
            )
        return list(session.scalars(statement))

    def get_with_provenance(self, session: Session, identifier: uuid.UUID) -> Chunk | None:
        """Return one chunk and its ordered source regions."""

        statement = (
            select(Chunk)
            .where(Chunk.id == identifier)
            .options(selectinload(Chunk.provenance_records))
        )
        return session.scalar(statement)

    def list_page_for_run(
        self,
        session: Session,
        processing_run_id: uuid.UUID,
        *,
        chunk_role: str | None = None,
        kind: str | None = None,
        page_no: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        statement = (
            select(Chunk)
            .where(Chunk.processing_run_id == processing_run_id)
            .options(selectinload(Chunk.provenance_records))
            .order_by(Chunk.chunk_role, Chunk.ordinal)
        )
        statement = self._apply_filters(
            statement, chunk_role=chunk_role, kind=kind, page_no=page_no
        )
        return list(session.scalars(statement.offset(offset).limit(limit)))

    def count_filtered_for_run(
        self,
        session: Session,
        processing_run_id: uuid.UUID,
        *,
        chunk_role: str | None = None,
        kind: str | None = None,
        page_no: int | None = None,
    ) -> int:
        statement = select(func.count()).select_from(Chunk).where(
            Chunk.processing_run_id == processing_run_id
        )
        statement = self._apply_filters(
            statement, chunk_role=chunk_role, kind=kind, page_no=page_no
        )
        return int(session.scalar(statement) or 0)

    @staticmethod
    def _apply_filters(statement, *, chunk_role, kind, page_no):
        if chunk_role is not None:
            statement = statement.where(Chunk.chunk_role == chunk_role)
        if kind is not None:
            statement = statement.where(Chunk.kind == kind)
        if page_no is not None:
            statement = statement.where(
                Chunk.page_start.is_not(None),
                Chunk.page_end.is_not(None),
                Chunk.page_start <= page_no,
                Chunk.page_end >= page_no,
            )
        return statement

    def list_for_run_and_fingerprint(
        self, session: Session, processing_run_id: uuid.UUID, fingerprint: str
    ) -> list[Chunk]:
        """Return one generated chunk set for idempotency checks."""

        statement = (
            select(Chunk)
            .where(
                Chunk.processing_run_id == processing_run_id,
                Chunk.chunking_fingerprint == fingerprint,
            )
            .options(selectinload(Chunk.provenance_records))
            .order_by(Chunk.chunk_role, Chunk.ordinal)
        )
        return list(session.scalars(statement))

    def count_for_run(self, session: Session, processing_run_id: uuid.UUID) -> int:
        """Count chunks belonging to a processing run."""

        statement = select(func.count()).select_from(Chunk).where(
            Chunk.processing_run_id == processing_run_id
        )
        return int(session.scalar(statement) or 0)

    def delete_for_run(self, session: Session, processing_run_id: uuid.UUID) -> None:
        """Remove prior chunks when a queued job is safely retried."""

        session.execute(delete(Chunk).where(Chunk.processing_run_id == processing_run_id))


class ArtifactRepository(BaseRepository[Artifact]):
    """Persistence helpers for filesystem artifacts."""

    def __init__(self) -> None:
        super().__init__(Artifact)

    def list_for_run(self, session: Session, processing_run_id: uuid.UUID) -> list[Artifact]:
        """Return artifacts for a processing run."""

        statement = (
            select(Artifact)
            .where(Artifact.processing_run_id == processing_run_id)
            .order_by(Artifact.created_at)
        )
        return list(session.scalars(statement))

    def count_for_run(self, session: Session, processing_run_id: uuid.UUID) -> int:
        """Count artifacts belonging to a processing run."""

        statement = select(func.count()).select_from(Artifact).where(
            Artifact.processing_run_id == processing_run_id
        )
        return int(session.scalar(statement) or 0)

    def get_for_run_type(
        self,
        session: Session,
        processing_run_id: uuid.UUID,
        artifact_type: str,
        *,
        page_no: int | None = None,
        doc_item_ref: str | None = None,
    ) -> Artifact | None:
        """Resolve one authoritative artifact by stored run metadata."""

        statement = select(Artifact).where(
            Artifact.processing_run_id == processing_run_id,
            Artifact.artifact_type == artifact_type,
        )
        if page_no is not None:
            statement = statement.where(Artifact.page_no == page_no)
        if doc_item_ref is not None:
            statement = statement.where(Artifact.doc_item_ref == doc_item_ref)
        return session.scalar(statement.order_by(Artifact.created_at.desc()).limit(1))

    def get_overlay_by_fingerprint(
        self, session: Session, processing_run_id: uuid.UUID, fingerprint: str
    ) -> Artifact | None:
        """Return a cached evidence overlay with an exact fingerprint."""

        statement = (
            select(Artifact)
            .where(
                Artifact.processing_run_id == processing_run_id,
                Artifact.artifact_type == "evidence_overlay",
                Artifact.metadata_json["overlay_fingerprint"].astext == fingerprint,
            )
            .limit(1)
        )
        return session.scalar(statement)

    def delete_for_run(self, session: Session, processing_run_id: uuid.UUID) -> None:
        """Remove prior artifact records when a queued job is safely retried."""

        session.execute(delete(Artifact).where(Artifact.processing_run_id == processing_run_id))


class ProvenanceRepository(BaseRepository[ProvenanceRecord]):
    """Persistence helpers for exact chunk-to-source regions."""

    def __init__(self) -> None:
        super().__init__(ProvenanceRecord)

    def list_for_chunk(self, session: Session, chunk_id: uuid.UUID) -> list[ProvenanceRecord]:
        """Return deterministic source-region order for one chunk."""

        statement = (
            select(ProvenanceRecord)
            .where(ProvenanceRecord.chunk_id == chunk_id)
            .order_by(
                ProvenanceRecord.page_no,
                ProvenanceRecord.doc_item_ref,
                ProvenanceRecord.bbox_top,
                ProvenanceRecord.bbox_left,
            )
        )
        return list(session.scalars(statement))

    def count_for_run(self, session: Session, processing_run_id: uuid.UUID) -> int:
        """Count source regions belonging to one processing run."""

        statement = select(func.count()).select_from(ProvenanceRecord).where(
            ProvenanceRecord.processing_run_id == processing_run_id
        )
        return int(session.scalar(statement) or 0)


class QueryRunRepository(BaseRepository[QueryRun]):
    """Persistence helpers for query runs."""

    def __init__(self) -> None:
        super().__init__(QueryRun)


class RetrievalHitRepository(BaseRepository[RetrievalHit]):
    """Persistence helpers for ranked retrieval hits."""

    def __init__(self) -> None:
        super().__init__(RetrievalHit)

    def list_for_query(self, session: Session, query_run_id: uuid.UUID) -> list[RetrievalHit]:
        """Return persisted hits in rank order."""

        statement = (
            select(RetrievalHit)
            .where(RetrievalHit.query_run_id == query_run_id)
            .order_by(RetrievalHit.rank)
        )
        return list(session.scalars(statement))


class EvaluationRunRepository(BaseRepository[EvaluationRun]):
    """Persistence helpers for evaluation runs."""

    def __init__(self) -> None:
        super().__init__(EvaluationRun)

    def list_for_document(
        self, session: Session, document_id: uuid.UUID
    ) -> list[EvaluationRun]:
        """Return evaluation runs for a document, newest first."""

        statement = (
            select(EvaluationRun)
            .where(EvaluationRun.document_id == document_id)
            .order_by(EvaluationRun.created_at.desc())
        )
        return list(session.scalars(statement))
