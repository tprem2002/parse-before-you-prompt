"""In-process baseline and Docling processing lifecycles for background tasks."""

from __future__ import annotations

import uuid
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pymupdf
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.enums import (
    ArtifactType,
    ChunkKind,
    ChunkRole,
    IndexMode,
    PipelineType,
    ProcessingStatus,
)
from app.core.logging import get_logger
from app.db.models import Artifact, Chunk, ProcessingRun, ProvenanceRecord
from app.db.repositories import (
    ArtifactRepository,
    ChunkRepository,
    DocumentRepository,
    ProcessingRunRepository,
)
from app.db.session import SessionLocal
from app.services.baseline_chunker import chunk_baseline, resolve_tokenizer
from app.services.baseline_parser import PAGE_SEPARATOR_TEMPLATE, parse_pdf
from app.services.file_storage import FileStorage
from app.services.docling_converter import (
    convert_docling_standard,
    docling_standard_configuration,
    export_docling_artifacts,
)
from app.services.docling_chunker import generate_chunks
from app.services.embedding_index_service import index_processing_run


logger = get_logger(__name__)


class ProcessingInterrupted(RuntimeError):
    """Raised at a persisted stage boundary during graceful shutdown."""


def _canonical_fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _worker_configuration(
    configuration: dict[str, Any],
    index_mode: IndexMode,
    *,
    request_id: str | None = None,
    retry_of: uuid.UUID | None = None,
    resume_action: str | None = None,
) -> dict[str, Any]:
    """Persist restart-safe worker intent without treating it as parser configuration."""

    queued_at = datetime.now(timezone.utc).isoformat()
    return {
        **configuration,
        "background_processing": {
            "implementation": "in_process_single_worker_v1",
            "index_mode": index_mode.value,
            "queued_at": queued_at,
            "request_id": request_id,
            "retry_of": str(retry_of) if retry_of else None,
            "resume_action": resume_action,
            "stage_history": [
                {"stage": "queued", "progress_percent": 0, "at": queued_at}
            ],
        },
        "processing_configuration_fingerprint": _canonical_fingerprint(configuration),
    }


def _pipeline_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """Remove worker/index results before validating local parser configuration."""

    return {
        key: value
        for key, value in configuration.items()
        if key not in {
            "background_processing",
            "embedding_indexing",
            "processing_configuration_fingerprint",
        }
    }


def _persist_skipped_indexing(run: ProcessingRun) -> None:
    configuration = dict(run.configuration_json or {})
    configuration["embedding_indexing"] = {
        "status": "skipped",
        "reason": "index_mode_skip",
        "azure_request_executed": False,
        "chroma_write_executed": False,
    }
    run.configuration_json = configuration


def _index_mode_for(run: ProcessingRun) -> IndexMode:
    raw = (run.configuration_json or {}).get("background_processing")
    value = raw.get("index_mode") if isinstance(raw, dict) else IndexMode.AUTO.value
    try:
        return IndexMode(value)
    except ValueError:
        return IndexMode.AUTO


def _transition(
    run_id: uuid.UUID,
    stage: str,
    progress_percent: int,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Persist a monotonic stage boundary in a short transaction."""

    if stop_requested is not None and stop_requested():
        raise ProcessingInterrupted("Worker shutdown requested")
    with SessionLocal.begin() as session:
        run = session.get(ProcessingRun, run_id, with_for_update=True)
        if run is None:
            raise LookupError(f"Processing run not found: {run_id}")
        if run.status != ProcessingStatus.RUNNING.value:
            raise ProcessingInterrupted(f"Run is no longer active: {run.status}")
        run.stage = stage
        run.progress_percent = max(run.progress_percent, progress_percent)
        configuration = dict(run.configuration_json or {})
        background = configuration.get("background_processing")
        metadata = dict(background) if isinstance(background, dict) else {}
        metadata.setdefault("work_started_at", datetime.now(timezone.utc).isoformat())
        history = metadata.get("stage_history")
        stage_history = list(history) if isinstance(history, list) else []
        if not stage_history or stage_history[-1].get("stage") != stage:
            stage_history.append(
                {
                    "stage": stage,
                    "progress_percent": run.progress_percent,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
        metadata["stage_history"] = stage_history
        configuration["background_processing"] = metadata
        run.configuration_json = configuration


def _finish_failed(
    run_id: uuid.UUID,
    *,
    started: float,
    category: str,
    message: str,
    exception_type: str | None = None,
) -> None:
    """Persist only safe failure information while retaining the last progress value."""

    with SessionLocal.begin() as session:
        run = session.get(ProcessingRun, run_id, with_for_update=True)
        if run is None or run.status == ProcessingStatus.COMPLETED.value:
            return
        run.status = ProcessingStatus.FAILED.value
        run.stage = "failed"
        run.completed_at = datetime.now(timezone.utc)
        run.duration_ms = round((perf_counter() - started) * 1000)
        run.error_message = message
        configuration = dict(run.configuration_json or {})
        raw = configuration.get("background_processing")
        background = dict(raw) if isinstance(raw, dict) else {}
        background["failure"] = {
            "category": category,
            "exception_type": exception_type,
            "failed_at": run.completed_at.isoformat(),
        }
        history = background.get("stage_history")
        stage_history = list(history) if isinstance(history, list) else []
        stage_history.append(
            {
                "stage": "failed",
                "progress_percent": run.progress_percent,
                "at": run.completed_at.isoformat(),
            }
        )
        background["stage_history"] = stage_history
        configuration["background_processing"] = background
        run.configuration_json = configuration


def _finish_completed(run_id: uuid.UUID, *, started: float) -> None:
    with SessionLocal.begin() as session:
        run = session.get(ProcessingRun, run_id, with_for_update=True)
        if run is None:
            raise LookupError(f"Processing run not found: {run_id}")
        run.status = ProcessingStatus.COMPLETED.value
        run.stage = "completed"
        run.progress_percent = 100
        run.completed_at = datetime.now(timezone.utc)
        run.duration_ms = round((perf_counter() - started) * 1000)
        run.error_message = None
        configuration = dict(run.configuration_json or {})
        raw = configuration.get("background_processing")
        background = dict(raw) if isinstance(raw, dict) else {}
        history = background.get("stage_history")
        stage_history = list(history) if isinstance(history, list) else []
        stage_history.append(
            {
                "stage": "completed",
                "progress_percent": 100,
                "at": run.completed_at.isoformat(),
            }
        )
        background["stage_history"] = stage_history
        configuration["background_processing"] = background
        run.configuration_json = configuration


def _fully_indexed(session: Session, run: ProcessingRun) -> bool:
    vector_total = int(
        session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(
                Chunk.processing_run_id == run.id,
                Chunk.chunk_role == ChunkRole.VECTOR_INDEX.value,
            )
        )
        or 0
    )
    indexed_total = int(
        session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(
                Chunk.processing_run_id == run.id,
                Chunk.chunk_role == ChunkRole.VECTOR_INDEX.value,
                Chunk.vector_id.is_not(None),
            )
        )
        or 0
    )
    section = (run.configuration_json or {}).get("embedding_indexing")
    status = section.get("status") if isinstance(section, dict) else None
    return vector_total > 0 and indexed_total == vector_total and status in {
        "indexed",
        "reindexed",
        "reconciled",
    }


def _clone_local_outputs_for_indexing(
    session: Session,
    source: ProcessingRun,
    *,
    parser_configuration: dict[str, Any],
    index_mode: IndexMode,
    request_id: str | None,
) -> ProcessingRun:
    """Create a queued run by copying local outputs, without reparsing or conversion."""

    configuration = _worker_configuration(
        parser_configuration,
        index_mode,
        request_id=request_id,
        resume_action="index_only",
    )
    background = dict(configuration["background_processing"])
    background["reused_local_outputs_from"] = str(source.id)
    configuration["background_processing"] = background
    target = ProcessingRun(
        document_id=source.document_id,
        pipeline_type=source.pipeline_type,
        status=ProcessingStatus.QUEUED.value,
        stage="queued",
        progress_percent=0,
        configuration_json=configuration,
    )
    session.add(target)
    session.flush()
    for artifact in ArtifactRepository().list_for_run(session, source.id):
        if artifact.artifact_type == ArtifactType.EVIDENCE_OVERLAY.value:
            continue
        session.add(
            Artifact(
                processing_run_id=target.id,
                artifact_type=artifact.artifact_type,
                storage_path=artifact.storage_path,
                page_no=artifact.page_no,
                doc_item_ref=artifact.doc_item_ref,
                metadata_json=artifact.metadata_json,
            )
        )
    for chunk in ChunkRepository().list_for_run(session, source.id):
        cloned_id = uuid.uuid4()
        session.add(
            Chunk(
                id=cloned_id,
                processing_run_id=target.id,
                document_id=chunk.document_id,
                ordinal=chunk.ordinal,
                chunk_role=chunk.chunk_role,
                kind=chunk.kind,
                raw_text=chunk.raw_text,
                embedding_text=chunk.embedding_text,
                token_count=chunk.token_count,
                raw_token_count=chunk.raw_token_count,
                contextualized_token_count=chunk.contextualized_token_count,
                max_token_count=chunk.max_token_count,
                section_path=chunk.section_path,
                captions=chunk.captions,
                doc_item_refs=chunk.doc_item_refs,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                table_ref=chunk.table_ref,
                picture_ref=chunk.picture_ref,
                is_derived_content=chunk.is_derived_content,
                content_classification=chunk.content_classification,
                chunking_fingerprint=chunk.chunking_fingerprint,
                serializer_metadata=chunk.serializer_metadata,
                chunk_metadata=chunk.chunk_metadata,
                header_repetition_status=chunk.header_repetition_status,
                overflow=chunk.overflow,
                vector_collection=None,
                vector_id=None,
            )
        )
        session.add_all(
            [
                ProvenanceRecord(
                    chunk_id=cloned_id,
                    document_id=record.document_id,
                    processing_run_id=target.id,
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
            ]
        )
    session.flush()
    return target


def _clone_docling_artifacts_for_chunking(
    session: Session,
    source: ProcessingRun,
    *,
    parser_configuration: dict[str, Any],
    index_mode: IndexMode,
    request_id: str | None,
) -> ProcessingRun:
    """Queue chunking/indexing from compatible lossless artifacts without conversion."""

    configuration = _worker_configuration(
        parser_configuration,
        index_mode,
        request_id=request_id,
        resume_action="chunk_and_index",
    )
    background = dict(configuration["background_processing"])
    background["reused_local_outputs_from"] = str(source.id)
    configuration["background_processing"] = background
    target = ProcessingRun(
        document_id=source.document_id,
        pipeline_type=source.pipeline_type,
        status=ProcessingStatus.QUEUED.value,
        stage="queued",
        progress_percent=0,
        configuration_json=configuration,
    )
    session.add(target)
    session.flush()
    for artifact in ArtifactRepository().list_for_run(session, source.id):
        if artifact.artifact_type == ArtifactType.EVIDENCE_OVERLAY.value:
            continue
        session.add(
            Artifact(
                processing_run_id=target.id,
                artifact_type=artifact.artifact_type,
                storage_path=artifact.storage_path,
                page_no=artifact.page_no,
                doc_item_ref=artifact.doc_item_ref,
                metadata_json=artifact.metadata_json,
            )
        )
    session.flush()
    return target


def _execute_indexing_stage(
    run_id: uuid.UUID,
    index_mode: IndexMode,
    settings: Settings,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Apply skip/auto/required behavior without eager provider initialization."""

    if index_mode is IndexMode.SKIP:
        with SessionLocal.begin() as session:
            run = session.get(ProcessingRun, run_id, with_for_update=True)
            if run is None:
                raise LookupError(f"Processing run not found: {run_id}")
            _persist_skipped_indexing(run)
        return

    if not settings.azure_openai_embedding_ready:
        if index_mode is IndexMode.REQUIRED:
            raise RuntimeError("Required embedding configuration became unavailable")
        with SessionLocal.begin() as session:
            run = session.get(ProcessingRun, run_id, with_for_update=True)
            if run is None:
                raise LookupError(f"Processing run not found: {run_id}")
            configuration = dict(run.configuration_json or {})
            configuration["embedding_indexing"] = {
                "status": "skipped",
                "reason": "embedding_configuration_unavailable",
                "missing_settings": settings.azure_openai_embedding_missing_settings,
                "azure_request_executed": False,
                "chroma_write_executed": False,
            }
            raw_background = configuration.get("background_processing")
            background = dict(raw_background) if isinstance(raw_background, dict) else {}
            warnings = background.get("warnings")
            safe_warnings = list(warnings) if isinstance(warnings, list) else []
            safe_warnings.append(
                "Indexing was skipped because embedding configuration is unavailable."
            )
            background["warnings"] = list(dict.fromkeys(safe_warnings))
            configuration["background_processing"] = background
            run.configuration_json = configuration
        return

    def stage_callback(stage: str) -> None:
        progress = 78 if stage == "embedding" else 92
        _transition(
            run_id,
            stage,
            progress,
            stop_requested=stop_requested,
        )

    index_processing_run(
        run_id,
        execute=True,
        allow_running=True,
        settings=settings,
        stage_callback=stage_callback,
    )


@dataclass(frozen=True, slots=True)
class BaselineRunSelection:
    """The run selected for a request and whether background work is needed."""

    run: ProcessingRun
    reused: bool
    should_schedule: bool


@dataclass(frozen=True, slots=True)
class DoclingRunSelection:
    """The Docling run selected for a request and whether work is needed."""

    run: ProcessingRun
    reused: bool
    should_schedule: bool


def baseline_configuration(settings: Settings | None = None) -> dict[str, Any]:
    """Return the exact configuration persisted for baseline comparisons."""

    settings = settings or get_settings()
    tokenizer = resolve_tokenizer(settings)
    return {
        "configuration_version": 1,
        "pipeline_type": PipelineType.BASELINE.value,
        "parser": "pymupdf_plain_text",
        "pymupdf_version": pymupdf.VersionBind,
        "text_extraction": {"mode": "text", "sort": False, "ocr": False},
        "page_separator": PAGE_SEPARATOR_TEMPLATE.strip(),
        "tokenizer": {
            "model_hint": tokenizer.model_hint,
            "fallback_encoding": tokenizer.fallback_encoding,
            "resolved_encoding": tokenizer.encoding.name,
            "used_fallback": tokenizer.used_fallback,
        },
        "chunk_max_tokens": settings.chunk_max_tokens,
        "chunk_overlap_tokens": settings.baseline_chunk_overlap_tokens,
        "chunk_role": ChunkRole.VECTOR_INDEX.value,
        "chunk_kind": ChunkKind.TEXT.value,
    }


def select_or_create_baseline_run(
    session: Session,
    document_id: uuid.UUID,
    settings: Settings | None = None,
    *,
    index_mode: IndexMode = IndexMode.AUTO,
    force_reprocess: bool = False,
    retry_failed: bool = False,
    request_id: str | None = None,
) -> BaselineRunSelection:
    """Serialize run selection and reuse active or identical completed work."""

    settings = settings or get_settings()
    document_repository = DocumentRepository()
    run_repository = ProcessingRunRepository()
    chunk_repository = ChunkRepository()
    artifact_repository = ArtifactRepository()

    document = document_repository.get_for_update(session, document_id)
    if document is None:
        raise LookupError("Document not found")

    parser_configuration = baseline_configuration(settings)
    active = run_repository.get_active_baseline(session, document_id)
    if active is not None:
        if _pipeline_configuration(active.configuration_json or {}) != parser_configuration:
            raise RuntimeError("An incompatible baseline processing run is already active")
        if _index_mode_for(active) is not index_mode:
            raise RuntimeError("A baseline run with a different index_mode is already active")
        return BaselineRunSelection(
            run=active,
            reused=True,
            should_schedule=active.status == ProcessingStatus.QUEUED.value,
        )

    candidates = list(
        session.scalars(
            select(ProcessingRun)
            .where(
                ProcessingRun.document_id == document_id,
                ProcessingRun.pipeline_type == PipelineType.BASELINE.value,
            )
            .order_by(ProcessingRun.completed_at.desc().nullslast())
        )
    )
    compatible = [
        candidate
        for candidate in candidates
        if _pipeline_configuration(candidate.configuration_json or {}) == parser_configuration
    ]
    completed = next(
        (
            candidate
            for candidate in compatible
            if candidate.status == ProcessingStatus.COMPLETED.value
            and chunk_repository.count_for_run(session, candidate.id) > 0
            and artifact_repository.count_for_run(session, candidate.id) >= 2
        ),
        None,
    )
    if completed is not None and not force_reprocess:
        wants_index = index_mode is IndexMode.REQUIRED or (
            index_mode is IndexMode.AUTO and settings.azure_openai_embedding_ready
        )
        if wants_index and not _fully_indexed(session, completed):
            resumed = _clone_local_outputs_for_indexing(
                session,
                completed,
                parser_configuration=parser_configuration,
                index_mode=index_mode,
                request_id=request_id,
            )
            return BaselineRunSelection(run=resumed, reused=True, should_schedule=True)
        return BaselineRunSelection(run=completed, reused=True, should_schedule=False)

    failed = next(
        (
            candidate
            for candidate in compatible
            if candidate.status == ProcessingStatus.FAILED.value
        ),
        None,
    )
    if failed is not None and not force_reprocess and not retry_failed:
        return BaselineRunSelection(run=failed, reused=True, should_schedule=False)

    configuration = _worker_configuration(
        parser_configuration,
        index_mode,
        request_id=request_id,
        retry_of=failed.id if failed is not None and retry_failed else None,
    )
    run = ProcessingRun(
        document_id=document.id,
        pipeline_type=PipelineType.BASELINE.value,
        status=ProcessingStatus.QUEUED.value,
        stage="queued",
        progress_percent=0,
        configuration_json=configuration,
    )
    run_repository.add(session, run)
    return BaselineRunSelection(run=run, reused=False, should_schedule=True)


def execute_baseline_run(
    run_id: uuid.UUID,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Execute baseline work with short database transactions at stage boundaries."""

    started = perf_counter()
    settings = get_settings()
    storage = FileStorage(settings)
    try:
        with SessionLocal() as session:
            run = session.get(ProcessingRun, run_id)
            if run is None or run.status != ProcessingStatus.RUNNING.value:
                return
            document = run.document
            document_id = document.id
            source_path = Path(document.storage_path)
            source_storage_path = document.storage_path
            source_filename = document.filename
            source_sha256 = document.sha256
            source_size = document.file_size_bytes
            index_mode = _index_mode_for(run)

        _transition(run_id, "parsing", 20, stop_requested=stop_requested)
        parsed = parse_pdf(source_path)

        _transition(run_id, "exporting_artifacts", 45, stop_requested=stop_requested)
        baseline_text_path = storage.write_baseline_text(
            document_id=document_id,
            processing_run_id=run_id,
            text=parsed.full_text,
        )

        _transition(run_id, "fixed_chunking", 65, stop_requested=stop_requested)
        baseline_chunks = chunk_baseline(parsed, settings)
        if not baseline_chunks:
            raise RuntimeError("Baseline extraction produced no chunkable text")

        with SessionLocal.begin() as session:
            run = session.get(ProcessingRun, run_id, with_for_update=True)
            if run is None or run.status != ProcessingStatus.RUNNING.value:
                raise ProcessingInterrupted("Run stopped before output persistence")
            run.document.page_count = parsed.page_count
            ChunkRepository().delete_for_run(session, run.id)
            ArtifactRepository().delete_for_run(session, run.id)
            session.add_all(
                [
                    Artifact(
                        processing_run_id=run.id,
                        artifact_type=ArtifactType.ORIGINAL_PDF.value,
                        storage_path=source_storage_path,
                        metadata_json={
                            "filename": source_filename,
                            "sha256": source_sha256,
                            "file_size_bytes": source_size,
                        },
                    ),
                    Artifact(
                        processing_run_id=run.id,
                        artifact_type=ArtifactType.BASELINE_TEXT.value,
                        storage_path=str(baseline_text_path),
                        metadata_json={
                            "encoding": "utf-8",
                            "page_count": parsed.page_count,
                            "character_count": len(parsed.full_text),
                            "page_separator": PAGE_SEPARATOR_TEMPLATE.strip(),
                        },
                    ),
                ]
            )
            session.add_all(
                [
                    Chunk(
                        processing_run_id=run.id,
                        document_id=document_id,
                        ordinal=chunk.ordinal,
                        chunk_role=ChunkRole.VECTOR_INDEX.value,
                        kind=ChunkKind.TEXT.value,
                        raw_text=chunk.text,
                        embedding_text=chunk.text,
                        token_count=chunk.token_count,
                        raw_token_count=chunk.token_count,
                        contextualized_token_count=chunk.token_count,
                        max_token_count=settings.chunk_max_tokens,
                        section_path=[],
                        captions=[],
                        doc_item_refs=[],
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        table_ref=None,
                        picture_ref=None,
                        is_derived_content=False,
                        content_classification="source",
                        chunking_fingerprint=None,
                        serializer_metadata={"serializer": "baseline_exact_text"},
                        chunk_metadata={
                            "chunker": "fixed_tiktoken",
                            "overlap_tokens": settings.baseline_chunk_overlap_tokens,
                        },
                        header_repetition_status=None,
                        overflow=False,
                        vector_collection=None,
                        vector_id=None,
                    )
                    for chunk in baseline_chunks
                ]
            )

        _execute_indexing_stage(
            run_id,
            index_mode,
            settings,
            stop_requested=stop_requested,
        )
        _finish_completed(run_id, started=started)
        logger.info(
            "Completed baseline run %s for document %s with %d chunks",
            run_id,
            document_id,
            len(baseline_chunks),
        )
    except ProcessingInterrupted:
        _finish_failed(
            run_id,
            started=started,
            category="process_interrupted",
            message="Processing stopped at a safe stage boundary during shutdown.",
            exception_type="ProcessingInterrupted",
        )
    except Exception as exc:
        logger.exception("Baseline processing run %s failed", run_id)
        _finish_failed(
            run_id,
            started=started,
            category="processing_failure",
            message="Baseline processing failed. Inspect server logs with the run ID.",
            exception_type=type(exc).__name__,
        )


def _completed_docling_artifacts_are_valid(
    session: Session,
    run: ProcessingRun,
    expected_page_count: int | None,
) -> bool:
    """Ensure a reusable completed run still has its required registered files."""

    artifacts = ArtifactRepository().list_for_run(session, run.id)
    types = {artifact.artifact_type for artifact in artifacts}
    required = {
        ArtifactType.ORIGINAL_PDF.value,
        ArtifactType.DOCLING_JSON.value,
        ArtifactType.MARKDOWN.value,
        ArtifactType.CONVERSION_MANIFEST.value,
    }
    page_images = sum(
        artifact.artifact_type == ArtifactType.PAGE_IMAGE.value for artifact in artifacts
    )
    return (
        required.issubset(types)
        and expected_page_count is not None
        and page_images == expected_page_count
        and all(Path(artifact.storage_path).is_file() for artifact in artifacts)
    )


def select_or_create_docling_standard_run(
    session: Session,
    document_id: uuid.UUID,
    settings: Settings | None = None,
    *,
    index_mode: IndexMode = IndexMode.AUTO,
    force_reprocess: bool = False,
    retry_failed: bool = False,
    request_id: str | None = None,
) -> DoclingRunSelection:
    """Reuse only exact, complete Docling work for this source hash and configuration."""

    settings = settings or get_settings()
    document_repository = DocumentRepository()
    run_repository = ProcessingRunRepository()
    document = document_repository.get_for_update(session, document_id)
    if document is None:
        raise LookupError("Document not found")

    parser_configuration = docling_standard_configuration(document.sha256, settings)
    active = session.scalar(
        select(ProcessingRun)
        .where(
            ProcessingRun.document_id == document.id,
            ProcessingRun.pipeline_type == PipelineType.DOCLING_STANDARD.value,
            ProcessingRun.status.in_(
                (ProcessingStatus.QUEUED.value, ProcessingStatus.RUNNING.value)
            ),
        )
        .order_by(ProcessingRun.started_at.desc().nullsfirst())
        .limit(1)
    )
    if active is not None:
        if _pipeline_configuration(active.configuration_json or {}) != parser_configuration:
            raise RuntimeError("An incompatible Docling processing run is already active")
        if _index_mode_for(active) is not index_mode:
            raise RuntimeError("A Docling run with a different index_mode is already active")
        return DoclingRunSelection(
            run=active,
            reused=True,
            should_schedule=active.status == ProcessingStatus.QUEUED.value,
        )

    candidates = list(
        session.scalars(
            select(ProcessingRun)
            .where(
                ProcessingRun.document_id == document.id,
                ProcessingRun.pipeline_type == PipelineType.DOCLING_STANDARD.value,
            )
            .order_by(ProcessingRun.completed_at.desc().nullslast())
        )
    )
    compatible = [
        candidate
        for candidate in candidates
        if _pipeline_configuration(candidate.configuration_json or {}) == parser_configuration
    ]
    artifact_complete = next(
        (
            candidate
            for candidate in compatible
            if candidate.status == ProcessingStatus.COMPLETED.value
            and _completed_docling_artifacts_are_valid(
                session, candidate, document.page_count
            )
        ),
        None,
    )
    completed = next(
        (
            candidate
            for candidate in compatible
            if candidate.status == ProcessingStatus.COMPLETED.value
            and _completed_docling_artifacts_are_valid(
                session, candidate, document.page_count
            )
            and ChunkRepository().count_for_run(session, candidate.id) > 0
        ),
        None,
    )
    if completed is not None and not force_reprocess:
        wants_index = index_mode is IndexMode.REQUIRED or (
            index_mode is IndexMode.AUTO and settings.azure_openai_embedding_ready
        )
        if wants_index and not _fully_indexed(session, completed):
            resumed = _clone_local_outputs_for_indexing(
                session,
                completed,
                parser_configuration=parser_configuration,
                index_mode=index_mode,
                request_id=request_id,
            )
            return DoclingRunSelection(run=resumed, reused=True, should_schedule=True)
        return DoclingRunSelection(run=completed, reused=True, should_schedule=False)
    if artifact_complete is not None and not force_reprocess:
        resumed = _clone_docling_artifacts_for_chunking(
            session,
            artifact_complete,
            parser_configuration=parser_configuration,
            index_mode=index_mode,
            request_id=request_id,
        )
        return DoclingRunSelection(run=resumed, reused=True, should_schedule=True)

    failed = next(
        (
            candidate
            for candidate in compatible
            if candidate.status == ProcessingStatus.FAILED.value
        ),
        None,
    )
    if failed is not None and not force_reprocess and not retry_failed:
        return DoclingRunSelection(run=failed, reused=True, should_schedule=False)

    configuration = _worker_configuration(
        parser_configuration,
        index_mode,
        request_id=request_id,
        retry_of=failed.id if failed is not None and retry_failed else None,
    )
    run = ProcessingRun(
        document_id=document.id,
        pipeline_type=PipelineType.DOCLING_STANDARD.value,
        status=ProcessingStatus.QUEUED.value,
        stage="queued",
        progress_percent=0,
        configuration_json=configuration,
    )
    run_repository.add(session, run)
    return DoclingRunSelection(run=run, reused=False, should_schedule=True)


def execute_docling_standard_run(
    run_id: uuid.UUID,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Execute local Docling conversion without holding a session during model work."""

    started = perf_counter()
    settings = get_settings()
    storage = FileStorage(settings)
    try:
        with SessionLocal() as session:
            run = session.get(ProcessingRun, run_id)
            if run is None or run.status != ProcessingStatus.RUNNING.value:
                return
            document = run.document
            document_id = document.id
            source_path = Path(document.storage_path)
            source_filename = document.filename
            source_sha256 = document.sha256
            configuration = dict(run.configuration_json or {})
            started_at = run.started_at or datetime.now(timezone.utc)
            index_mode = _index_mode_for(run)
        expected = docling_standard_configuration(source_sha256, settings)
        if _pipeline_configuration(configuration) != expected:
            raise RuntimeError("Queued Docling configuration no longer matches current settings")

        _transition(run_id, "parsing", 10, stop_requested=stop_requested)
        conversion = convert_docling_standard(source_path, settings)

        _transition(run_id, "exporting_artifacts", 45, stop_requested=stop_requested)
        exported = export_docling_artifacts(
            conversion=conversion,
            storage=storage,
            document_id=document_id,
            processing_run_id=run_id,
            source_filename=source_filename,
            source_sha256=source_sha256,
            source_path=source_path,
            configuration=configuration,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            duration_ms=round((perf_counter() - started) * 1000),
        )

        from docling.datamodel.base_models import ConversionStatus

        with SessionLocal.begin() as session:
            run = session.get(ProcessingRun, run_id, with_for_update=True)
            if run is None or run.status != ProcessingStatus.RUNNING.value:
                raise ProcessingInterrupted("Run stopped before artifact persistence")
            ArtifactRepository().delete_for_run(session, run.id)
            session.add_all(
                [
                    Artifact(
                        processing_run_id=run.id,
                        artifact_type=artifact.artifact_type,
                        storage_path=artifact.storage_path,
                        page_no=artifact.page_no,
                        doc_item_ref=artifact.doc_item_ref,
                        metadata_json=artifact.metadata or {},
                    )
                    for artifact in exported.artifacts
                ]
            )
            if conversion.result.status in {
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            }:
                run.document.page_count = len(conversion.result.document.pages)

        chunking = None
        if conversion.result.status is ConversionStatus.SUCCESS:
            if stop_requested is not None and stop_requested():
                raise ProcessingInterrupted("Worker shutdown requested")
            chunking = generate_chunks(
                run_id,
                allow_running=True,
                persist_progress=True,
                settings=settings,
            )
            if chunking.warnings:
                with SessionLocal.begin() as session:
                    run = session.get(ProcessingRun, run_id, with_for_update=True)
                    if run is not None:
                        run_configuration = dict(run.configuration_json or {})
                        raw_background = run_configuration.get("background_processing")
                        background = (
                            dict(raw_background) if isinstance(raw_background, dict) else {}
                        )
                        background["warnings"] = list(chunking.warnings)
                        run_configuration["background_processing"] = background
                        run.configuration_json = run_configuration
            _execute_indexing_stage(
                run_id,
                index_mode,
                settings,
                stop_requested=stop_requested,
            )

        final_timestamp = datetime.now(timezone.utc)
        exported.manifest["completed_at"] = final_timestamp.isoformat()
        exported.manifest["duration_ms"] = round((perf_counter() - started) * 1000)
        if chunking is not None:
            exported.manifest["chunking"] = chunking.as_dict()
            exported.manifest["background_processing"] = {
                "index_mode": index_mode.value,
                "single_worker": True,
            }
        storage.write_artifact_text(
            document_id=document_id,
            processing_run_id=run_id,
            relative_path="conversion-manifest.json",
            text=json.dumps(exported.manifest, ensure_ascii=False, indent=2),
        )

        if conversion.result.status is ConversionStatus.SUCCESS:
            _finish_completed(run_id, started=started)
            final_status = ProcessingStatus.COMPLETED.value
        else:
            with SessionLocal.begin() as session:
                run = session.get(ProcessingRun, run_id, with_for_update=True)
                if run is None:
                    raise LookupError(f"Processing run not found: {run_id}")
                run.completed_at = final_timestamp
                run.duration_ms = round((perf_counter() - started) * 1000)
                run.progress_percent = 100
                if conversion.result.status is ConversionStatus.PARTIAL_SUCCESS:
                    run.status = ProcessingStatus.PARTIAL.value
                    run.stage = "partial_conversion"
                    run.error_message = "Docling reported partial success; inspect safe warnings."
                else:
                    run.status = ProcessingStatus.FAILED.value
                    run.stage = "failed"
                    run.error_message = "Docling conversion did not complete successfully."
                final_status = run.status
        logger.info(
            "Finished Docling standard run %s status=%s artifacts=%d",
            run_id,
            final_status,
            len(exported.artifacts),
        )
    except ProcessingInterrupted:
        _finish_failed(
            run_id,
            started=started,
            category="process_interrupted",
            message="Processing stopped at a safe stage boundary during shutdown.",
            exception_type="ProcessingInterrupted",
        )
    except Exception as exc:
        logger.exception("Docling standard processing run %s failed", run_id)
        _finish_failed(
            run_id,
            started=started,
            category="processing_failure",
            message="Docling processing failed. Inspect server logs with the run ID.",
            exception_type=type(exc).__name__,
        )


def execute_processing_run(
    run_id: uuid.UUID,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> None:
    """Dispatch one persisted run without allowing concurrent pipeline execution."""

    with SessionLocal() as session:
        run = session.get(ProcessingRun, run_id)
        if run is None:
            logger.warning("Ignoring missing queued processing run %s", run_id)
            return
        pipeline_type = run.pipeline_type
        index_mode = _index_mode_for(run)
        raw_background = (run.configuration_json or {}).get("background_processing")
        resume_action = (
            raw_background.get("resume_action")
            if isinstance(raw_background, dict)
            else None
        )

    if resume_action == "index_only":
        started = perf_counter()
        try:
            _execute_indexing_stage(
                run_id,
                index_mode,
                get_settings(),
                stop_requested=stop_requested,
            )
            _finish_completed(run_id, started=started)
        except ProcessingInterrupted:
            _finish_failed(
                run_id,
                started=started,
                category="process_interrupted",
                message="Indexing stopped at a safe stage boundary during shutdown.",
                exception_type="ProcessingInterrupted",
            )
        except Exception as exc:
            logger.exception("Index-only processing run %s failed", run_id)
            _finish_failed(
                run_id,
                started=started,
                category="indexing_failure",
                message="Indexing failed. Inspect server logs with the run ID.",
                exception_type=type(exc).__name__,
            )
        return

    if resume_action == "chunk_and_index":
        started = perf_counter()
        try:
            settings = get_settings()
            chunking = generate_chunks(
                run_id,
                allow_running=True,
                persist_progress=True,
                settings=settings,
            )
            if chunking.warnings:
                with SessionLocal.begin() as session:
                    run = session.get(ProcessingRun, run_id, with_for_update=True)
                    if run is not None:
                        configuration = dict(run.configuration_json or {})
                        raw_background = configuration.get("background_processing")
                        background = (
                            dict(raw_background)
                            if isinstance(raw_background, dict)
                            else {}
                        )
                        background["warnings"] = list(chunking.warnings)
                        configuration["background_processing"] = background
                        run.configuration_json = configuration
            _execute_indexing_stage(
                run_id,
                index_mode,
                settings,
                stop_requested=stop_requested,
            )
            _finish_completed(run_id, started=started)
        except ProcessingInterrupted:
            _finish_failed(
                run_id,
                started=started,
                category="process_interrupted",
                message="Chunking stopped at a safe stage boundary during shutdown.",
                exception_type="ProcessingInterrupted",
            )
        except Exception as exc:
            logger.exception("Chunk-and-index processing run %s failed", run_id)
            _finish_failed(
                run_id,
                started=started,
                category="processing_failure",
                message="Chunking or indexing failed. Inspect server logs with the run ID.",
                exception_type=type(exc).__name__,
            )
        return

    if pipeline_type == PipelineType.BASELINE.value:
        execute_baseline_run(run_id, stop_requested=stop_requested)
    elif pipeline_type == PipelineType.DOCLING_STANDARD.value:
        execute_docling_standard_run(run_id, stop_requested=stop_requested)
    else:
        with SessionLocal() as session:
            run = session.get(ProcessingRun, run_id)
            if run is not None:
                run.status = ProcessingStatus.FAILED.value
                run.stage = "failed"
                run.completed_at = datetime.now(timezone.utc)
                run.error_message = "The queued processing pipeline is not supported."
                session.commit()
