"""Strict external-vector retrieval with PostgreSQL-authoritative evidence."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import Settings, get_settings
from app.core.enums import ArtifactType, ChunkRole, PipelineType, ProcessingStatus
from app.core.errors import ApplicationError, ConfigurationError
from app.db.models import Artifact, Chunk, ProcessingRun, ProvenanceRecord
from app.db.session import SessionLocal
from app.providers.embeddings.azure_openai import (
    PROVIDER_IMPLEMENTATION_VERSION,
    final_embedding_fingerprint,
    get_azure_openai_embedding_provider,
    input_representation_version_for,
)
from app.services.chroma_service import (
    ChromaQueryHit,
    collection_name_for,
    get_chroma_service,
)


ELIGIBLE_PIPELINES = {PipelineType.BASELINE.value, PipelineType.DOCLING_STANDARD.value}
DERIVED_VISUAL_DESCRIPTION_LABEL = (
    "Derived visual description — generated locally from the source image"
)


class RetrievalError(ApplicationError):
    """A retrieval operation failed safely."""


class RunNotIndexedError(RetrievalError):
    """The selected run is not a complete, compatible external-vector index."""


class RetrievalIntegrityError(RetrievalError):
    """Chroma and PostgreSQL disagree about one or more returned records."""


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    id: uuid.UUID
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


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    rank: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    processing_run_id: uuid.UUID
    pipeline_type: str
    distance: float
    kind: str
    embedding_text: str
    raw_text: str
    source_classification: str
    section_path: tuple[Any, ...]
    source_captions: tuple[str, ...]
    derived_visual_description: str | None
    page_start: int | None
    page_end: int | None
    doc_item_refs: tuple[str, ...]
    table_ref: str | None
    picture_ref: str | None
    precise_provenance_available: bool
    provenance_records: tuple[EvidenceProvenance, ...]
    evidence_overlay_available: bool
    cached_overlay_available: bool
    evidence_image_pages: tuple[int, ...]
    contextualized_token_count: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    processing_run_id: uuid.UUID
    document_id: uuid.UUID
    pipeline_type: str
    requested_top_k: int
    actual_hit_count: int
    chroma_hit_count: int
    collection_chunk_count: int
    collection: str
    embedding_fingerprint: str
    input_representation_version: str
    query_vector_dimension: int
    query_embedding_duration_ms: int
    chroma_search_duration_ms: int
    postgres_resolution_duration_ms: int
    total_duration_ms: int
    total_contextualized_evidence_tokens: int
    token_budget_truncated: bool
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class RunRetrievalReadiness:
    processing_run_id: uuid.UUID
    document_id: uuid.UUID
    pipeline_type: str
    run_status: str
    eligible_pipeline: bool
    eligible_vector_chunk_count: int
    excluded_chunk_count: int
    indexed_chunk_count: int
    fully_indexed: bool
    expected_collection: str | None
    embedding_fingerprint: str | None
    vector_dimension: int | None
    input_representation_version: str
    indexing_issues: tuple[str, ...]

    def as_dict(self, settings: Settings, *, requested_top_k: int) -> dict[str, object]:
        return {
            "processing_run_id": str(self.processing_run_id),
            "document_id": str(self.document_id),
            "pipeline_type": self.pipeline_type,
            "run_status": self.run_status,
            "eligible_pipeline": self.eligible_pipeline,
            "eligible_vector_index_chunk_count": self.eligible_vector_chunk_count,
            "excluded_chunk_count": self.excluded_chunk_count,
            "indexed_chunk_count": self.indexed_chunk_count,
            "fully_indexed": self.fully_indexed,
            "expected_collection": self.expected_collection,
            "embedding_fingerprint": self.embedding_fingerprint,
            "vector_dimension": self.vector_dimension,
            "input_representation_version": self.input_representation_version,
            "indexing_issues": list(self.indexing_issues),
            "embedding_ready": settings.azure_openai_embedding_ready,
            "chat_ready": settings.azure_openai_chat_ready,
            "rag_model_ready": settings.rag_model_ready,
            "missing_embedding_settings": settings.azure_openai_embedding_missing_settings,
            "missing_chat_settings": settings.azure_openai_chat_missing_settings,
            "requested_top_k": requested_top_k,
            "model_call_will_occur": False,
            "chroma_query_will_occur": False,
        }


def _indexing_section(run: ProcessingRun) -> dict[str, Any]:
    section = (run.configuration_json or {}).get("embedding_indexing")
    return dict(section) if isinstance(section, dict) else {}


def inspect_run_readiness(
    processing_run_id: uuid.UUID,
    *,
    settings: Settings | None = None,
) -> RunRetrievalReadiness:
    """Inspect only PostgreSQL state; never initialize providers or contact Chroma."""

    configured_settings = settings or get_settings()
    with SessionLocal() as session:
        run = session.get(ProcessingRun, processing_run_id)
        if run is None:
            raise LookupError(f"Processing run not found: {processing_run_id}")
        all_chunks = list(
            session.scalars(
                select(Chunk)
                .where(Chunk.processing_run_id == processing_run_id)
                .order_by(Chunk.chunk_role, Chunk.ordinal, Chunk.id)
            )
        )

    vector_chunks = [
        chunk for chunk in all_chunks if chunk.chunk_role == ChunkRole.VECTOR_INDEX.value
    ]
    section = _indexing_section(run)
    expected_version = (
        input_representation_version_for(run.pipeline_type)
        if run.pipeline_type in ELIGIBLE_PIPELINES
        else "unsupported"
    )
    issues: list[str] = []
    if run.status != ProcessingStatus.COMPLETED.value:
        issues.append("processing_run_not_completed")
    if run.pipeline_type not in ELIGIBLE_PIPELINES:
        issues.append("pipeline_not_eligible")
    if not vector_chunks:
        issues.append("no_vector_index_chunks")

    collection = section.get("chroma_collection")
    fingerprint = section.get("final_embedding_fingerprint")
    vector_dimension = section.get("vector_dimension")
    indexed_count = sum(
        chunk.vector_id is not None and chunk.vector_collection is not None
        for chunk in vector_chunks
    )
    if section.get("status") != "indexed":
        issues.append("indexing_status_not_indexed")
    if not isinstance(collection, str) or not collection:
        issues.append("missing_chroma_collection")
        collection = None
    if not isinstance(fingerprint, str) or not fingerprint:
        issues.append("missing_final_embedding_fingerprint")
        fingerprint = None
    if not isinstance(vector_dimension, int) or vector_dimension < 1:
        issues.append("missing_vector_dimension")
        vector_dimension = None
    if section.get("input_representation_version") != expected_version:
        issues.append("input_representation_version_mismatch")
    if section.get("embedding_provider_version") != PROVIDER_IMPLEMENTATION_VERSION:
        issues.append("embedding_provider_version_mismatch")
    if section.get("indexed_chunk_count") != len(vector_chunks):
        issues.append("indexed_chunk_count_mismatch")
    if indexed_count != len(vector_chunks):
        issues.append("partial_vector_metadata")
    collections = {chunk.vector_collection for chunk in vector_chunks}
    if len(collections) != 1 or None in collections:
        issues.append("inconsistent_chunk_collections")
    elif collection is not None and collections != {collection}:
        issues.append("chunk_collection_mismatch")
    if any(
        chunk.vector_id is not None and chunk.vector_id != str(chunk.id)
        for chunk in vector_chunks
    ):
        issues.append("non_deterministic_vector_id")
    if collection is not None and fingerprint is not None:
        if collection != collection_name_for(run.pipeline_type, fingerprint):
            issues.append("collection_name_identity_mismatch")

    return RunRetrievalReadiness(
        processing_run_id=run.id,
        document_id=run.document_id,
        pipeline_type=run.pipeline_type,
        run_status=run.status,
        eligible_pipeline=run.pipeline_type in ELIGIBLE_PIPELINES,
        eligible_vector_chunk_count=len(vector_chunks),
        excluded_chunk_count=len(all_chunks) - len(vector_chunks),
        indexed_chunk_count=indexed_count,
        fully_indexed=not issues,
        expected_collection=collection,
        embedding_fingerprint=fingerprint,
        vector_dimension=vector_dimension,
        input_representation_version=expected_version,
        indexing_issues=tuple(issues),
    )


def _precise(record: ProvenanceRecord) -> bool:
    return all(
        value is not None
        for value in (
            record.bbox_left,
            record.bbox_top,
            record.bbox_right,
            record.bbox_bottom,
            record.coordinate_origin,
        )
    )


def _caption_texts(values: list[Any]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = value.get("text") if isinstance(value, dict) else value
        if isinstance(text, str) and text.strip() and text.strip() not in result:
            result.append(text.strip())
    return tuple(result)


def _reference_texts(values: list[Any]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            text = str(value.get("self_ref") or value.get("doc_item_ref") or "")
        else:
            text = str(value)
        if text.strip() and text.strip() not in result:
            result.append(text.strip())
    return tuple(result)


def _derived_description(chunk: Chunk) -> str | None:
    descriptions: list[str] = []
    components = (chunk.chunk_metadata or {}).get("picture_components") or []
    for component in components:
        if not isinstance(component, dict):
            continue
        generated = component.get("generated_description")
        if isinstance(generated, dict):
            text = generated.get("text")
            label = generated.get("label")
            if (
                isinstance(text, str)
                and text.strip()
                and label == DERIVED_VISUAL_DESCRIPTION_LABEL
            ):
                descriptions.append(text.strip())
    return "\n".join(descriptions) or None


def _validate_metadata(
    hit: ChromaQueryHit,
    chunk: Chunk,
    *,
    readiness: RunRetrievalReadiness,
) -> None:
    expected: dict[str, object] = {
        "chunk_id": str(chunk.id),
        "document_id": str(readiness.document_id),
        "processing_run_id": str(readiness.processing_run_id),
        "pipeline_type": readiness.pipeline_type,
        "chunk_role": ChunkRole.VECTOR_INDEX.value,
        "kind": chunk.kind,
        "embedding_fingerprint": readiness.embedding_fingerprint,
        "input_representation_version": readiness.input_representation_version,
        "embedding_provider_version": PROVIDER_IMPLEMENTATION_VERSION,
    }
    mismatches = [key for key, value in expected.items() if hit.metadata.get(key) != value]
    if mismatches:
        raise RetrievalIntegrityError(
            "Chroma metadata disagrees with PostgreSQL for returned vector "
            f"{hit.vector_id}: {', '.join(sorted(mismatches))}"
        )


def _build_evidence(
    hit: ChromaQueryHit,
    chunk: Chunk,
    *,
    rank: int,
    readiness: RunRetrievalReadiness,
    artifacts: list[Artifact],
) -> Evidence:
    if chunk.processing_run_id != readiness.processing_run_id:
        raise RetrievalIntegrityError(f"Returned chunk {chunk.id} belongs to a foreign run")
    if chunk.document_id != readiness.document_id:
        raise RetrievalIntegrityError(f"Returned chunk {chunk.id} belongs to a foreign document")
    if chunk.chunk_role != ChunkRole.VECTOR_INDEX.value:
        raise RetrievalIntegrityError(f"Returned chunk {chunk.id} is not a vector_index chunk")
    if chunk.vector_id != hit.vector_id:
        raise RetrievalIntegrityError(f"Returned vector ID does not match chunk {chunk.id}")
    if chunk.vector_collection != readiness.expected_collection:
        raise RetrievalIntegrityError(f"Returned chunk {chunk.id} has a foreign collection")
    if hit.document != chunk.embedding_text:
        raise RetrievalIntegrityError(
            f"Stored Chroma document text does not match PostgreSQL chunk {chunk.id}"
        )
    if not math.isfinite(hit.distance):
        raise RetrievalIntegrityError(f"Returned vector {hit.vector_id} has a non-finite distance")
    _validate_metadata(hit, chunk, readiness=readiness)

    provenance = tuple(
        EvidenceProvenance(
            id=record.id,
            doc_item_ref=record.doc_item_ref,
            page_no=record.page_no,
            bbox_left=record.bbox_left,
            bbox_top=record.bbox_top,
            bbox_right=record.bbox_right,
            bbox_bottom=record.bbox_bottom,
            coordinate_origin=record.coordinate_origin,
            char_start=record.char_start,
            char_end=record.char_end,
            evidence_role=record.evidence_role,
        )
        for record in chunk.provenance_records
    )
    precise_pages = tuple(sorted({record.page_no for record in chunk.provenance_records if _precise(record)}))
    page_image_pages = {
        artifact.page_no
        for artifact in artifacts
        if artifact.artifact_type == ArtifactType.PAGE_IMAGE.value and artifact.page_no is not None
    }
    has_docling_json = any(
        artifact.artifact_type == ArtifactType.DOCLING_JSON.value for artifact in artifacts
    )
    overlay_available = bool(
        precise_pages
        and has_docling_json
        and all(page in page_image_pages for page in precise_pages)
    )
    cached_overlay = any(
        artifact.artifact_type == ArtifactType.EVIDENCE_OVERLAY.value
        and str((artifact.metadata_json or {}).get("chunk_id")) == str(chunk.id)
        for artifact in artifacts
    )
    return Evidence(
        evidence_id=f"E{rank}",
        rank=rank,
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        processing_run_id=chunk.processing_run_id,
        pipeline_type=readiness.pipeline_type,
        distance=hit.distance,
        kind=chunk.kind,
        embedding_text=chunk.embedding_text,
        raw_text=chunk.raw_text,
        source_classification=chunk.content_classification,
        section_path=tuple(chunk.section_path),
        source_captions=_caption_texts(chunk.captions),
        derived_visual_description=_derived_description(chunk),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        doc_item_refs=_reference_texts(chunk.doc_item_refs),
        table_ref=chunk.table_ref,
        picture_ref=chunk.picture_ref,
        precise_provenance_available=bool(precise_pages),
        provenance_records=provenance,
        evidence_overlay_available=overlay_available,
        cached_overlay_available=cached_overlay,
        evidence_image_pages=precise_pages,
        contextualized_token_count=chunk.contextualized_token_count,
    )


def retrieve(
    processing_run_id: uuid.UUID,
    question: str,
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    """Embed one question, search the exact persisted collection, and validate every hit."""

    started = perf_counter()
    configured_settings = settings or get_settings()
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question must not be empty")
    requested_top_k = top_k if top_k is not None else configured_settings.rag_top_k
    if requested_top_k < 1 or requested_top_k > configured_settings.rag_max_top_k:
        raise ValueError(
            f"top_k must be between 1 and {configured_settings.rag_max_top_k}"
        )

    readiness = inspect_run_readiness(processing_run_id, settings=configured_settings)
    if not readiness.fully_indexed:
        raise RunNotIndexedError(
            "Processing run is not fully indexed: " + ", ".join(readiness.indexing_issues)
        )
    if not configured_settings.azure_openai_embedding_ready:
        missing = ", ".join(configured_settings.azure_openai_embedding_missing_settings)
        raise ConfigurationError(f"Azure OpenAI embeddings are not configured: {missing}")
    assert readiness.expected_collection is not None
    assert readiness.embedding_fingerprint is not None
    assert readiness.vector_dimension is not None

    with SessionLocal() as session:
        run = session.get(ProcessingRun, processing_run_id)
        assert run is not None
        section = _indexing_section(run)
    persisted_service_model = (
        str(section["service_returned_model"])
        if section.get("service_returned_model") is not None
        else None
    )
    expected_fingerprint, _ = final_embedding_fingerprint(
        configured_settings,
        pipeline_type=readiness.pipeline_type,
        service_model=persisted_service_model,
        vector_dimension=readiness.vector_dimension,
    )
    if expected_fingerprint != readiness.embedding_fingerprint:
        raise RetrievalIntegrityError(
            "Current embedding identity does not match the indexed collection"
        )

    chroma = get_chroma_service(configured_settings)
    collection = chroma.get_validated_collection(
        name=readiness.expected_collection,
        pipeline_type=readiness.pipeline_type,
        embedding_fingerprint=readiness.embedding_fingerprint,
        vector_dimension=readiness.vector_dimension,
        input_representation_version=readiness.input_representation_version,
        embedding_provider_version=PROVIDER_IMPLEMENTATION_VERSION,
    )
    collection_chunk_count = collection.count()

    embedding_started = perf_counter()
    query_embedding = get_azure_openai_embedding_provider(
        configured_settings
    ).embed_query_result(normalized_question)
    query_embedding_duration_ms = round((perf_counter() - embedding_started) * 1000)
    if query_embedding.deployment_name != section.get("deployment_name"):
        raise RetrievalIntegrityError(
            "Query embedding deployment does not match the indexed collection"
        )
    query_fingerprint, _ = final_embedding_fingerprint(
        configured_settings,
        pipeline_type=readiness.pipeline_type,
        service_model=query_embedding.service_model,
        vector_dimension=query_embedding.vector_dimension,
    )
    if query_fingerprint != readiness.embedding_fingerprint:
        raise RetrievalIntegrityError(
            "Query embedding response identity does not match the indexed collection"
        )
    query_vector = list(query_embedding.vectors[0])
    try:
        query_vector = [float(value) for value in query_vector]
    except (TypeError, ValueError) as exc:
        raise RetrievalIntegrityError("Query embedding contains a non-numeric value") from exc
    if not query_vector or not all(math.isfinite(value) for value in query_vector):
        raise RetrievalIntegrityError("Query embedding is empty or contains non-finite values")
    if len(query_vector) != readiness.vector_dimension:
        raise RetrievalIntegrityError(
            "Query embedding dimension does not match the indexed collection"
        )

    search_started = perf_counter()
    hits = chroma.query_by_vector(
        collection,
        query_vector=query_vector,
        n_results=requested_top_k,
        where={
            "$and": [
                {"document_id": {"$eq": str(readiness.document_id)}},
                {"processing_run_id": {"$eq": str(readiness.processing_run_id)}},
                {"pipeline_type": {"$eq": readiness.pipeline_type}},
            ]
        },
    )
    chroma_search_duration_ms = round((perf_counter() - search_started) * 1000)
    if len({hit.vector_id for hit in hits}) != len(hits):
        raise RetrievalIntegrityError("Chroma returned duplicate vector IDs")

    resolution_started = perf_counter()
    vector_ids = [hit.vector_id for hit in hits]
    with SessionLocal() as session:
        chunks = list(
            session.scalars(
                select(Chunk)
                .where(Chunk.vector_id.in_(vector_ids))
                .options(selectinload(Chunk.provenance_records))
            )
        ) if vector_ids else []
        artifacts = list(
            session.scalars(
                select(Artifact).where(Artifact.processing_run_id == processing_run_id)
            )
        )
    chunks_by_vector_id = {
        chunk.vector_id: chunk for chunk in chunks if chunk.vector_id is not None
    }
    if len(chunks_by_vector_id) != len(vector_ids):
        raise RetrievalIntegrityError(
            "One or more Chroma vector IDs did not resolve uniquely in PostgreSQL"
        )
    validated = tuple(
        _build_evidence(
            hit,
            chunks_by_vector_id[hit.vector_id],
            rank=rank,
            readiness=readiness,
            artifacts=artifacts,
        )
        for rank, hit in enumerate(hits, start=1)
    )
    postgres_resolution_duration_ms = round((perf_counter() - resolution_started) * 1000)

    selected: list[Evidence] = []
    selected_tokens = 0
    for evidence in validated:
        if selected_tokens + evidence.contextualized_token_count > configured_settings.rag_max_evidence_tokens:
            break
        selected.append(evidence)
        selected_tokens += evidence.contextualized_token_count

    return RetrievalResult(
        processing_run_id=readiness.processing_run_id,
        document_id=readiness.document_id,
        pipeline_type=readiness.pipeline_type,
        requested_top_k=requested_top_k,
        actual_hit_count=len(selected),
        chroma_hit_count=len(validated),
        collection_chunk_count=collection_chunk_count,
        collection=readiness.expected_collection,
        embedding_fingerprint=readiness.embedding_fingerprint,
        input_representation_version=readiness.input_representation_version,
        query_vector_dimension=len(query_vector),
        query_embedding_duration_ms=query_embedding_duration_ms,
        chroma_search_duration_ms=chroma_search_duration_ms,
        postgres_resolution_duration_ms=postgres_resolution_duration_ms,
        total_duration_ms=round((perf_counter() - started) * 1000),
        total_contextualized_evidence_tokens=selected_tokens,
        token_budget_truncated=len(selected) < len(validated),
        evidence=tuple(selected),
    )
