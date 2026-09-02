"""Offline planning and explicit Azure-to-Chroma indexing for completed runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.enums import ChunkRole, PipelineType, ProcessingStatus
from app.core.errors import ApplicationError
from app.db.models import Chunk, ProcessingRun
from app.db.session import SessionLocal
from app.providers.embeddings.azure_openai import (
    PROVIDER_NAME,
    PROVIDER_IMPLEMENTATION_VERSION,
    final_embedding_fingerprint,
    get_azure_openai_embedding_provider,
    input_representation_version_for,
    provisional_embedding_fingerprint,
)
from app.providers.embeddings.base import (
    EmbeddingBatchPlan,
    EmbeddingConfigurationError,
    build_embedding_batch_plan,
)
from app.services.baseline_chunker import resolve_tokenizer
from app.services.chroma_service import (
    ChromaIndexError,
    ChromaRecord,
    ChromaService,
    collection_name_for,
    get_chroma_service,
)


ELIGIBLE_PIPELINES = {PipelineType.BASELINE.value, PipelineType.DOCLING_STANDARD.value}


class IndexingConflictError(ApplicationError):
    """Existing vector state conflicts with the requested semantic identity."""


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Safe dry-run or execution result; never contains source text or vectors."""

    processing_run_id: uuid.UUID
    pipeline_type: str
    status: str
    execute: bool
    force_reindex: bool
    execution_possible: bool
    missing_configuration: tuple[str, ...]
    eligible_chunk_count: int
    excluded_chunk_count: int
    already_indexed_chunk_count: int
    aggregate_tokens: int
    minimum_input_tokens: int
    maximum_input_tokens: int
    token_count_mismatches: tuple[dict[str, object], ...]
    planned_batches: tuple[dict[str, object], ...]
    tokenizer_model_hint: str
    tokenizer_encoding: str
    tokenizer_fallback_used: bool
    input_representation_version: str
    provisional_fingerprint: str
    final_fingerprint: str | None = None
    collection_name: str | None = None
    collection_created: bool | None = None
    vector_dimension: int | None = None
    provider_name: str | None = None
    deployment_name: str | None = None
    service_model: str | None = None
    request_duration_ms: int | None = None
    indexing_duration_ms: int | None = None
    usage_prompt_tokens: int | None = None
    usage_total_tokens: int | None = None
    request_ids: tuple[str, ...] = ()
    retry_count: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Serialize only safe operational metadata for manual commands."""

        return {
            "processing_run_id": str(self.processing_run_id),
            "pipeline_type": self.pipeline_type,
            "status": self.status,
            "execute": self.execute,
            "force_reindex": self.force_reindex,
            "configuration": {
                "embedding_ready": self.execution_possible,
                "missing_embedding_settings": list(self.missing_configuration),
            },
            "chunks": {
                "eligible": self.eligible_chunk_count,
                "excluded": self.excluded_chunk_count,
                "already_indexed": self.already_indexed_chunk_count,
            },
            "tokens": {
                "aggregate": self.aggregate_tokens,
                "minimum": self.minimum_input_tokens,
                "maximum": self.maximum_input_tokens,
                "mismatches": list(self.token_count_mismatches),
            },
            "tokenizer": {
                "model_hint": self.tokenizer_model_hint,
                "resolved_encoding": self.tokenizer_encoding,
                "fallback_used": self.tokenizer_fallback_used,
            },
            "input_representation_version": self.input_representation_version,
            "planned_batches": list(self.planned_batches),
            "embedding": {
                "provider": self.provider_name,
                "deployment_name": self.deployment_name,
                "service_model": self.service_model,
                "provisional_fingerprint": self.provisional_fingerprint,
                "final_fingerprint": self.final_fingerprint,
                "vector_dimension": self.vector_dimension,
                "request_duration_ms": self.request_duration_ms,
                "usage_prompt_tokens": self.usage_prompt_tokens,
                "usage_total_tokens": self.usage_total_tokens,
                "request_ids": list(self.request_ids),
                "retry_count": self.retry_count,
            },
            "chroma": {
                "collection_name": self.collection_name,
                "collection_created": self.collection_created,
            },
            "indexing_duration_ms": self.indexing_duration_ms,
        }


@dataclass(frozen=True, slots=True)
class _RunInputs:
    run_id: uuid.UUID
    document_id: uuid.UUID
    pipeline_type: str
    configuration: dict[str, Any]
    chunks: tuple[Chunk, ...]
    excluded_count: int


def _load_inputs(processing_run_id: uuid.UUID, *, allow_running: bool = False) -> _RunInputs:
    with SessionLocal() as session:
        run = session.get(ProcessingRun, processing_run_id)
        if run is None:
            raise LookupError(f"Processing run not found: {processing_run_id}")
        allowed_statuses = {ProcessingStatus.COMPLETED.value}
        if allow_running:
            allowed_statuses.add(ProcessingStatus.RUNNING.value)
        if run.status not in allowed_statuses:
            raise EmbeddingConfigurationError(
                f"Processing run {processing_run_id} is not eligible for indexing in status "
                f"{run.status}"
            )
        if run.pipeline_type not in ELIGIBLE_PIPELINES:
            raise EmbeddingConfigurationError(
                f"Pipeline {run.pipeline_type} is not eligible for vector indexing"
            )
        all_chunks = list(
            session.scalars(
                select(Chunk)
                .where(Chunk.processing_run_id == processing_run_id)
                .order_by(Chunk.chunk_role, Chunk.ordinal, Chunk.id)
            )
        )
        chunks = tuple(
            sorted(
                (chunk for chunk in all_chunks if chunk.chunk_role == ChunkRole.VECTOR_INDEX.value),
                key=lambda chunk: (chunk.ordinal, str(chunk.id)),
            )
        )
        if not chunks:
            raise EmbeddingConfigurationError("Processing run has no vector_index chunks")
        expected_ordinals = list(range(len(chunks)))
        if [chunk.ordinal for chunk in chunks] != expected_ordinals:
            raise EmbeddingConfigurationError(
                "Vector-index chunk ordinals must be unique, contiguous, and deterministic"
            )
        for chunk in chunks:
            if chunk.processing_run_id != run.id or chunk.document_id != run.document_id:
                raise EmbeddingConfigurationError(
                    f"Chunk {chunk.id} does not belong to the selected processing run"
                )
            if not chunk.embedding_text.strip():
                raise EmbeddingConfigurationError(f"Chunk {chunk.id} has empty embedding_text")
            if chunk.contextualized_token_count < 0:
                raise EmbeddingConfigurationError(f"Chunk {chunk.id} has an invalid token count")
            if (chunk.vector_id is None) != (chunk.vector_collection is None):
                raise IndexingConflictError(
                    f"Chunk {chunk.id} has incomplete existing vector metadata"
                )
            if chunk.vector_id is not None and chunk.vector_id != str(chunk.id):
                raise IndexingConflictError(
                    f"Chunk {chunk.id} has a non-deterministic existing vector ID"
                )
        return _RunInputs(
            run_id=run.id,
            document_id=run.document_id,
            pipeline_type=run.pipeline_type,
            configuration=dict(run.configuration_json or {}),
            chunks=chunks,
            excluded_count=len(all_chunks) - len(chunks),
        )


def _build_plan(inputs: _RunInputs, settings: Settings) -> EmbeddingBatchPlan:
    tokenizer = resolve_tokenizer(settings)
    return build_embedding_batch_plan(
        [chunk.embedding_text for chunk in inputs.chunks],
        input_ids=[str(chunk.id) for chunk in inputs.chunks],
        stored_token_counts=[chunk.contextualized_token_count for chunk in inputs.chunks],
        encoding=tokenizer.encoding,
        tokenizer_model_hint=tokenizer.model_hint,
        tokenizer_fallback_used=tokenizer.used_fallback,
        max_inputs=settings.embedding_batch_max_inputs,
        max_tokens=settings.embedding_batch_max_tokens,
        per_input_max_tokens=settings.chunk_max_tokens,
    )


def _base_result(
    inputs: _RunInputs,
    plan: EmbeddingBatchPlan,
    settings: Settings,
    *,
    status: str,
    execute: bool,
    force_reindex: bool,
    provisional_fingerprint: str,
    **values: Any,
) -> IndexingResult:
    values.setdefault("provider_name", PROVIDER_NAME)
    values.setdefault("deployment_name", settings.azure_openai_embedding_deployment)
    values.setdefault(
        "input_representation_version",
        input_representation_version_for(inputs.pipeline_type),
    )
    indexed_count = sum(chunk.vector_id is not None for chunk in inputs.chunks)
    mismatches = tuple(
        {
            "chunk_id": mismatch.input_id,
            "stored_contextualized_tokens": mismatch.stored_count,
            "live_embedding_tokens": mismatch.live_count,
        }
        for mismatch in plan.token_count_mismatches
    )
    return IndexingResult(
        processing_run_id=inputs.run_id,
        pipeline_type=inputs.pipeline_type,
        status=status,
        execute=execute,
        force_reindex=force_reindex,
        execution_possible=settings.azure_openai_embedding_ready,
        missing_configuration=tuple(settings.azure_openai_embedding_missing_settings),
        eligible_chunk_count=len(inputs.chunks),
        excluded_chunk_count=inputs.excluded_count,
        already_indexed_chunk_count=indexed_count,
        aggregate_tokens=plan.aggregate_tokens,
        minimum_input_tokens=min(plan.token_counts),
        maximum_input_tokens=max(plan.token_counts),
        token_count_mismatches=mismatches,
        planned_batches=tuple(batch.as_dict() for batch in plan.batches),
        tokenizer_model_hint=plan.tokenizer_model_hint,
        tokenizer_encoding=plan.tokenizer_encoding,
        tokenizer_fallback_used=plan.tokenizer_fallback_used,
        provisional_fingerprint=provisional_fingerprint,
        **values,
    )


def _section_path_text(section_path: list[Any]) -> str:
    parts: list[str] = []
    for item in section_path:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = str(item.get("text") or item.get("title") or item.get("name") or "")
        else:
            value = str(item)
        if value.strip():
            parts.append(value.strip())
    return " > ".join(parts)


def _chroma_metadata(
    chunk: Chunk,
    *,
    pipeline_type: str,
    embedding_fingerprint: str,
    input_representation_version: str,
) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "chunk_id": str(chunk.id),
        "document_id": str(chunk.document_id),
        "processing_run_id": str(chunk.processing_run_id),
        "pipeline_type": pipeline_type,
        "chunk_role": chunk.chunk_role,
        "kind": chunk.kind,
        "source_classification": chunk.content_classification,
        "embedding_fingerprint": embedding_fingerprint,
        "input_representation_version": input_representation_version,
        "embedding_provider_version": PROVIDER_IMPLEMENTATION_VERSION,
    }
    optional: dict[str, str | int | None] = {
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_path_text": _section_path_text(chunk.section_path),
        "table_ref": chunk.table_ref,
        "picture_ref": chunk.picture_ref,
        "chunking_fingerprint": chunk.chunking_fingerprint,
    }
    metadata.update({key: value for key, value in optional.items() if value not in {None, ""}})
    return metadata


def _record_failure(
    run_id: uuid.UUID,
    *,
    provisional_fingerprint: str,
    category: str,
    error_type: str,
) -> None:
    """Record only safe failure diagnostics while retaining any valid prior index."""

    with SessionLocal.begin() as session:
        run = session.scalar(select(ProcessingRun).where(ProcessingRun.id == run_id).with_for_update())
        if run is None:
            return
        configuration = dict(run.configuration_json or {})
        section = dict(configuration.get("embedding_indexing") or {})
        section["last_attempt"] = {
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_category": category,
            "error_type": error_type,
            "provisional_fingerprint": provisional_fingerprint,
        }
        configuration["embedding_indexing"] = section
        run.configuration_json = configuration


def _persist_success(
    inputs: _RunInputs,
    *,
    collection_name: str,
    provisional_fingerprint: str,
    final_fingerprint: str,
    final_metadata: dict[str, object],
    vector_dimension: int,
    service_model: str | None,
    plan: EmbeddingBatchPlan,
    request_duration_ms: int,
    indexing_duration_ms: int,
    request_ids: tuple[str, ...],
    retry_count: int | None,
    settings: Settings,
    force_reindex: bool,
) -> None:
    expected_ids = [chunk.id for chunk in inputs.chunks]
    with SessionLocal.begin() as session:
        run = session.scalar(
            select(ProcessingRun).where(ProcessingRun.id == inputs.run_id).with_for_update()
        )
        if run is None or run.status != ProcessingStatus.COMPLETED.value:
            raise IndexingConflictError("Processing run changed before vector persistence")
        chunks = list(
            session.scalars(
                select(Chunk)
                .where(
                    Chunk.processing_run_id == inputs.run_id,
                    Chunk.chunk_role == ChunkRole.VECTOR_INDEX.value,
                )
                .order_by(Chunk.ordinal, Chunk.id)
                .with_for_update()
            )
        )
        if [chunk.id for chunk in chunks] != expected_ids:
            raise IndexingConflictError("Vector-index chunks changed before persistence")
        for chunk in chunks:
            if not force_reindex and chunk.vector_collection not in {None, collection_name}:
                raise IndexingConflictError(
                    f"Chunk {chunk.id} points to a different vector collection"
                )
            if not force_reindex and chunk.vector_id not in {None, str(chunk.id)}:
                raise IndexingConflictError(f"Chunk {chunk.id} has a conflicting vector ID")
            chunk.vector_collection = collection_name
            chunk.vector_id = str(chunk.id)

        configuration = dict(run.configuration_json or {})
        configuration["embedding_indexing"] = {
            "status": "indexed",
            "provider": PROVIDER_NAME,
            "embedding_provider_version": PROVIDER_IMPLEMENTATION_VERSION,
            "auth_mode": settings.azure_openai_auth_mode,
            "deployment_name": settings.azure_openai_embedding_deployment,
            "base_url_hash": final_metadata["normalized_base_url_hash"],
            "service_returned_model": service_model,
            "vector_dimension": vector_dimension,
            "tokenizer_model_hint": plan.tokenizer_model_hint,
            "tokenizer_encoding": plan.tokenizer_encoding,
            "tokenizer_fallback_used": plan.tokenizer_fallback_used,
            "input_representation_version": input_representation_version_for(
                inputs.pipeline_type
            ),
            "batch_limits": {
                "maximum_inputs": settings.embedding_batch_max_inputs,
                "maximum_aggregate_tokens": settings.embedding_batch_max_tokens,
                "maximum_input_tokens": settings.chunk_max_tokens,
            },
            "provisional_fingerprint": provisional_fingerprint,
            "final_embedding_fingerprint": final_fingerprint,
            "fingerprint_metadata": final_metadata,
            "chroma_collection": collection_name,
            "chroma_distance": "cosine",
            "indexed_chunk_count": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "indexing_duration_ms": indexing_duration_ms,
            "embedding_request_duration_ms": request_duration_ms,
            "request_ids": list(request_ids),
            "retry_count": retry_count,
            "last_attempt": {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "force_reindex": force_reindex,
            },
        }
        run.configuration_json = configuration


def _persist_reconciliation(
    inputs: _RunInputs,
    *,
    collection_name: str,
) -> None:
    with SessionLocal.begin() as session:
        chunks = list(
            session.scalars(
                select(Chunk)
                .where(
                    Chunk.processing_run_id == inputs.run_id,
                    Chunk.chunk_role == ChunkRole.VECTOR_INDEX.value,
                )
                .order_by(Chunk.ordinal, Chunk.id)
                .with_for_update()
            )
        )
        if [chunk.id for chunk in chunks] != [chunk.id for chunk in inputs.chunks]:
            raise IndexingConflictError("Vector-index chunks changed before reconciliation")
        for chunk in chunks:
            if chunk.vector_collection not in {None, collection_name}:
                raise IndexingConflictError(
                    f"Chunk {chunk.id} points to a different vector collection"
                )
            chunk.vector_collection = collection_name
            chunk.vector_id = str(chunk.id)


def _try_reconcile(
    inputs: _RunInputs,
    plan: EmbeddingBatchPlan,
    settings: Settings,
    chroma: ChromaService,
    *,
    provisional_fingerprint: str,
    started: float,
) -> IndexingResult | None:
    section = inputs.configuration.get("embedding_indexing")
    if not isinstance(section, dict):
        return None
    if section.get("provisional_fingerprint") != provisional_fingerprint:
        return None
    final_fingerprint = section.get("final_embedding_fingerprint")
    collection_name = section.get("chroma_collection")
    vector_dimension = section.get("vector_dimension")
    if not isinstance(final_fingerprint, str) or not isinstance(collection_name, str):
        return None
    if not isinstance(vector_dimension, int):
        return None
    try:
        collection = chroma.get_validated_collection(
            name=collection_name,
            pipeline_type=inputs.pipeline_type,
            embedding_fingerprint=final_fingerprint,
            vector_dimension=vector_dimension,
            input_representation_version=input_representation_version_for(
                inputs.pipeline_type
            ),
            embedding_provider_version=PROVIDER_IMPLEMENTATION_VERSION,
        )
    except ChromaIndexError as exc:
        if "not found" in str(exc).lower():
            return None
        raise
    expected_ids = [str(chunk.id) for chunk in inputs.chunks]
    if chroma.existing_ids(collection, expected_ids) != set(expected_ids):
        return None
    initially_indexed = sum(chunk.vector_id is not None for chunk in inputs.chunks)
    _persist_reconciliation(inputs, collection_name=collection_name)
    return _base_result(
        inputs,
        plan,
        settings,
        status="already_indexed" if initially_indexed == len(inputs.chunks) else "reconciled",
        execute=True,
        force_reindex=False,
        provisional_fingerprint=provisional_fingerprint,
        final_fingerprint=final_fingerprint,
        collection_name=collection_name,
        collection_created=False,
        vector_dimension=vector_dimension,
        provider_name=str(section.get("provider") or PROVIDER_NAME),
        deployment_name=str(section.get("deployment_name") or "") or None,
        service_model=(
            str(section["service_returned_model"])
            if section.get("service_returned_model") is not None
            else None
        ),
        indexing_duration_ms=round((perf_counter() - started) * 1000),
    )


def index_processing_run(
    processing_run_id: uuid.UUID,
    *,
    execute: bool,
    force_reindex: bool = False,
    allow_running: bool = False,
    stage_callback: Callable[[str], None] | None = None,
    settings: Settings | None = None,
) -> IndexingResult:
    """Plan or execute deterministic external-vector indexing for one completed run."""

    started = perf_counter()
    configured_settings = settings or get_settings()
    inputs = _load_inputs(processing_run_id, allow_running=allow_running)
    plan = _build_plan(inputs, configured_settings)
    provisional_fingerprint, _ = provisional_embedding_fingerprint(
        configured_settings,
        pipeline_type=inputs.pipeline_type,
    )

    if not execute:
        if force_reindex:
            raise EmbeddingConfigurationError("--force-reindex is only valid with --execute")
        return _base_result(
            inputs,
            plan,
            configured_settings,
            status="dry_run",
            execute=False,
            force_reindex=False,
            provisional_fingerprint=provisional_fingerprint,
            indexing_duration_ms=round((perf_counter() - started) * 1000),
        )

    if not configured_settings.azure_openai_embedding_ready:
        _record_failure(
            inputs.run_id,
            provisional_fingerprint=provisional_fingerprint,
            category="missing_configuration",
            error_type="EmbeddingConfigurationError",
        )
        missing = ", ".join(configured_settings.azure_openai_embedding_missing_settings)
        raise EmbeddingConfigurationError(f"Azure OpenAI embeddings are not configured: {missing}")

    existing_collections = {
        chunk.vector_collection for chunk in inputs.chunks if chunk.vector_collection is not None
    }
    section = inputs.configuration.get("embedding_indexing")
    existing_identity_matches = (
        isinstance(section, dict)
        and section.get("provisional_fingerprint") == provisional_fingerprint
    )
    if existing_collections and not existing_identity_matches and not force_reindex:
        raise IndexingConflictError(
            "Existing chunk vectors use a different or unrecorded embedding identity; "
            "use --force-reindex explicitly"
        )

    chroma = get_chroma_service(configured_settings)
    if not force_reindex:
        reconciled = _try_reconcile(
            inputs,
            plan,
            configured_settings,
            chroma,
            provisional_fingerprint=provisional_fingerprint,
            started=started,
        )
        if reconciled is not None:
            return reconciled

    collection = None
    collection_name: str | None = None
    collection_created = False
    newly_inserted_ids: list[str] = []
    try:
        if stage_callback is not None:
            stage_callback("embedding")
        provider = get_azure_openai_embedding_provider(configured_settings)
        embedding_result = provider.embed_documents(
            [chunk.embedding_text for chunk in inputs.chunks],
            input_ids=[str(chunk.id) for chunk in inputs.chunks],
        )
        final_fingerprint, final_metadata = final_embedding_fingerprint(
            configured_settings,
            pipeline_type=inputs.pipeline_type,
            service_model=embedding_result.service_model,
            vector_dimension=embedding_result.vector_dimension,
        )
        if stage_callback is not None:
            stage_callback("indexing")
        collection_name = collection_name_for(inputs.pipeline_type, final_fingerprint)
        base_url_hash = final_metadata["normalized_base_url_hash"]
        if not isinstance(base_url_hash, str):
            raise EmbeddingConfigurationError("Validated Azure base URL did not produce an identity hash")
        handle = chroma.ensure_collection(
            pipeline_type=inputs.pipeline_type,
            embedding_fingerprint=final_fingerprint,
            vector_dimension=embedding_result.vector_dimension,
            input_representation_version=input_representation_version_for(
                inputs.pipeline_type
            ),
            provider_name=embedding_result.provider_name,
            embedding_provider_version=PROVIDER_IMPLEMENTATION_VERSION,
            deployment_name=embedding_result.deployment_name,
            base_url_hash=base_url_hash,
            provisional_fingerprint=provisional_fingerprint,
        )
        collection = handle.collection
        collection_created = handle.created
        expected_ids = [str(chunk.id) for chunk in inputs.chunks]
        preexisting_ids = chroma.existing_ids(collection, expected_ids)
        newly_inserted_ids = sorted(set(expected_ids) - preexisting_ids)
        records = [
            ChromaRecord(
                vector_id=str(chunk.id),
                embedding=embedding_result.vectors[index],
                document=chunk.embedding_text,
                metadata=_chroma_metadata(
                    chunk,
                    pipeline_type=inputs.pipeline_type,
                    embedding_fingerprint=final_fingerprint,
                    input_representation_version=input_representation_version_for(
                        inputs.pipeline_type
                    ),
                ),
            )
            for index, chunk in enumerate(inputs.chunks)
        ]
        chroma.upsert_records(collection, records)
        indexing_duration_ms = round((perf_counter() - started) * 1000)
        _persist_success(
            inputs,
            collection_name=collection_name,
            provisional_fingerprint=provisional_fingerprint,
            final_fingerprint=final_fingerprint,
            final_metadata=final_metadata,
            vector_dimension=embedding_result.vector_dimension,
            service_model=embedding_result.service_model,
            plan=plan,
            request_duration_ms=embedding_result.request_duration_ms,
            indexing_duration_ms=indexing_duration_ms,
            request_ids=embedding_result.request_ids,
            retry_count=embedding_result.retry_count,
            settings=configured_settings,
            force_reindex=force_reindex,
        )
        return _base_result(
            inputs,
            plan,
            configured_settings,
            status="reindexed" if force_reindex else "indexed",
            execute=True,
            force_reindex=force_reindex,
            provisional_fingerprint=provisional_fingerprint,
            final_fingerprint=final_fingerprint,
            collection_name=collection_name,
            collection_created=collection_created,
            vector_dimension=embedding_result.vector_dimension,
            provider_name=embedding_result.provider_name,
            deployment_name=embedding_result.deployment_name,
            service_model=embedding_result.service_model,
            request_duration_ms=embedding_result.request_duration_ms,
            indexing_duration_ms=indexing_duration_ms,
            usage_prompt_tokens=embedding_result.usage_prompt_tokens,
            usage_total_tokens=embedding_result.usage_total_tokens,
            request_ids=embedding_result.request_ids,
            retry_count=embedding_result.retry_count,
        )
    except Exception as exc:
        if collection is not None and newly_inserted_ids:
            try:
                chroma.delete_ids(collection, newly_inserted_ids)
            except Exception:
                pass
        if collection_created and collection_name is not None:
            try:
                chroma.delete_collection_if_empty(collection_name)
            except Exception:
                pass
        _record_failure(
            inputs.run_id,
            provisional_fingerprint=provisional_fingerprint,
            category="indexing_failed",
            error_type=type(exc).__name__,
        )
        raise
