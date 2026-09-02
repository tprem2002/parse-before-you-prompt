"""Reusable, local-only Docling chunk generation for completed conversions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable

import tiktoken
from docling.chunking import HierarchicalChunker, HybridChunker
from docling_core.transforms.chunker import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
)
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.base import BaseDocSerializer, BaseSerializerProvider
from docling_core.transforms.serializer.markdown import MarkdownMetaSerializer
from docling_core.types.doc import (
    CodeItem,
    DescriptionMetaField,
    DoclingDocument,
    FloatingItem,
    FormulaItem,
    PictureItem,
    RefItem,
    TableItem,
)
from sqlalchemy import delete, select

from app.core.config import Settings, get_settings
from app.core.enums import (
    ArtifactType,
    ChunkKind,
    ChunkRole,
    ContentClassification,
    HeaderRepetitionStatus,
    PipelineType,
    ProcessingStatus,
)
from app.db.models import Chunk, ProcessingRun, ProvenanceRecord
from app.db.repositories import ArtifactRepository, ChunkRepository
from app.db.session import session_scope
from app.services.hashing import sha256_file
from app.services.provenance_service import ProvenanceRegion, extract_provenance


DERIVED_VISUAL_DESCRIPTION_LABEL = (
    "Derived visual description \u2014 generated locally from the source image"
)
CONTEXTUALIZED_TEXT_VERSION = "v1"
CHUNK_ID_NAMESPACE = uuid.UUID("f2b73870-106b-53e8-a9d4-f322571f30b3")
PROVENANCE_ID_NAMESPACE = uuid.UUID("4c22e08d-34f9-5ba1-9b6b-62e3cc3a242a")
ENABLED_DOCLING_PIPELINES = {PipelineType.DOCLING_STANDARD.value}


class SourceAwareMarkdownMetaSerializer(MarkdownMetaSerializer):
    """Label generated picture descriptions inside serialized chunk text."""

    def _serialize_meta_field(self, meta: Any, name: str, mark_meta: bool) -> str | None:
        value = getattr(meta, name, None)
        if isinstance(value, DescriptionMetaField):
            return f"{DERIVED_VISUAL_DESCRIPTION_LABEL}:\n{value.text}"
        return super()._serialize_meta_field(meta, name, mark_meta)


class Prompt7SerializerProvider(BaseSerializerProvider):
    """Create Docling's retrieval serializer with explicit derived labeling."""

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            meta_serializer=SourceAwareMarkdownMetaSerializer(),
        )


@dataclass(frozen=True, slots=True)
class TokenizerResolution:
    """Resolved tiktoken encoding and Docling adapter."""

    adapter: OpenAITokenizer
    encoding_name: str
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class ChunkingConfiguration:
    """Typed values that determine a Docling chunk set."""

    tokenizer_model_hint: str
    fallback_encoding: str
    max_tokens: int
    merge_peers: bool
    repeat_table_header: bool
    omit_header_on_overflow: bool
    contextualized_text_version: str = CONTEXTUALIZED_TEXT_VERSION

    @classmethod
    def from_settings(cls, settings: Settings) -> "ChunkingConfiguration":
        return cls(
            tokenizer_model_hint=settings.tiktoken_model_hint,
            fallback_encoding=settings.tiktoken_fallback_encoding,
            max_tokens=settings.chunk_max_tokens,
            merge_peers=settings.chunk_merge_peers,
            repeat_table_header=settings.chunk_repeat_table_header,
            omit_header_on_overflow=settings.chunk_omit_header_on_overflow,
        )


@dataclass(frozen=True, slots=True)
class ChunkStatistics:
    """Compact statistics for one persisted chunk role."""

    count: int
    raw_min: int
    raw_average: float
    raw_max: int
    contextualized_min: int
    contextualized_average: float
    contextualized_max: int
    overflow_count: int
    counts_by_kind: dict[str, int]
    with_headings: int
    with_captions: int
    missing_provenance: int


@dataclass(frozen=True, slots=True)
class ChunkGenerationResult:
    """Observable result returned by first-run and idempotent backfills."""

    processing_run_id: uuid.UUID
    docling_json_path: str
    docling_json_sha256: str
    configuration: dict[str, Any]
    configuration_fingerprint: str
    tokenizer_encoding: str
    tokenizer_fallback_used: bool
    hierarchical: ChunkStatistics
    hybrid: ChunkStatistics
    provenance_count: int
    warnings: tuple[str, ...]
    timings_ms: dict[str, int]
    created: bool
    reused: bool
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["processing_run_id"] = str(self.processing_run_id)
        return value


@dataclass(slots=True)
class ChunkDraft:
    """Fully validated chunk and its source regions before transaction writes."""

    id: uuid.UUID
    role: str
    ordinal: int
    kind: str
    raw_text: str
    embedding_text: str
    raw_token_count: int
    contextualized_token_count: int
    section_path: list[str]
    captions: list[dict[str, Any]]
    doc_item_refs: list[str]
    page_start: int | None
    page_end: int | None
    table_ref: str | None
    picture_ref: str | None
    content_classification: str
    serializer_metadata: dict[str, Any]
    chunk_metadata: dict[str, Any]
    header_repetition_status: str | None
    overflow: bool
    provenance: tuple[ProvenanceRegion, ...]


def resolve_tokenizer(
    configuration: ChunkingConfiguration,
) -> TokenizerResolution:
    """Resolve tiktoken explicitly and wrap it in Docling's OpenAI adapter."""

    fallback_used = False
    try:
        encoding = tiktoken.encoding_for_model(configuration.tokenizer_model_hint)
    except KeyError:
        encoding = tiktoken.get_encoding(configuration.fallback_encoding)
        fallback_used = True
    return TokenizerResolution(
        adapter=OpenAITokenizer(tokenizer=encoding, max_tokens=configuration.max_tokens),
        encoding_name=encoding.name,
        fallback_used=fallback_used,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _package_version(name: str) -> str:
    return importlib.metadata.version(name)


def _serializer_configuration() -> dict[str, Any]:
    return {
        "provider": "app.services.docling_chunker.Prompt7SerializerProvider",
        "document_serializer": (
            "docling_core.transforms.chunker.hierarchical_chunker.ChunkingDocSerializer"
        ),
        "text_serializer": "docling_core.transforms.serializer.markdown.MarkdownTextSerializer",
        "table_serializer": (
            "docling_core.transforms.chunker.hierarchical_chunker.TripletTableSerializer"
        ),
        "picture_serializer": (
            "docling_core.transforms.serializer.markdown.MarkdownPictureSerializer"
        ),
        "picture_meta_serializer": (
            "app.services.docling_chunker.SourceAwareMarkdownMetaSerializer"
        ),
        "formula_serializer": "MarkdownTextSerializer formula branch",
        "code_serializer": "MarkdownTextSerializer code branch",
    }


def build_configuration_payload(
    pipeline_type: str,
    configuration: ChunkingConfiguration,
    tokenizer: TokenizerResolution,
) -> dict[str, Any]:
    """Build the canonical, versioned input to the chunking fingerprint."""

    return {
        "docling_version": _package_version("docling"),
        "docling_core_version": _package_version("docling-core"),
        "pipeline_type": pipeline_type,
        "tokenizer_model_hint": configuration.tokenizer_model_hint,
        "tokenizer_encoding": tokenizer.encoding_name,
        "tokenizer_fallback_encoding": configuration.fallback_encoding,
        "tokenizer_fallback_used": tokenizer.fallback_used,
        "max_tokens": configuration.max_tokens,
        "merge_peers": configuration.merge_peers,
        "repeat_table_header": configuration.repeat_table_header,
        "omit_header_on_overflow": configuration.omit_header_on_overflow,
        "fixed_token_overlap": None,
        "contextualized_text_version": configuration.contextualized_text_version,
        "serializers": _serializer_configuration(),
        "source_derived_policy": {
            "source_captions_separate": True,
            "derived_descriptions_separate": True,
            "derived_description_label": DERIVED_VISUAL_DESCRIPTION_LABEL,
            "derived_picture_provenance_role": "derived_visual_anchor",
        },
    }


def _fingerprint(configuration: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()


def _captions(items: Iterable[Any], document: DoclingDocument) -> list[dict[str, Any]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        if getattr(getattr(item, "label", None), "value", None) == "caption":
            text = getattr(item, "text", "")
            if text:
                found[(item.self_ref, text)] = {
                    "text": text,
                    "doc_item_ref": item.self_ref,
                    "classification": "source",
                    "role": "human_authored_caption",
                }
        if isinstance(item, FloatingItem):
            for reference in item.captions:
                caption = reference.resolve(document)
                text = getattr(caption, "text", "")
                if text:
                    found[(caption.self_ref, text)] = {
                        "text": text,
                        "doc_item_ref": caption.self_ref,
                        "classification": "source",
                        "role": "human_authored_caption",
                    }
    return list(found.values())


_LABEL_TO_KIND = {
    "text": ChunkKind.TEXT.value,
    "paragraph": ChunkKind.TEXT.value,
    "section_header": ChunkKind.HEADING.value,
    "title": ChunkKind.HEADING.value,
    "list_item": ChunkKind.LIST.value,
    "table": ChunkKind.TABLE.value,
    "picture": ChunkKind.PICTURE.value,
    "caption": ChunkKind.CAPTION.value,
    "formula": ChunkKind.FORMULA.value,
    "code": ChunkKind.CODE.value,
    "footnote": ChunkKind.FOOTNOTE.value,
    "form": ChunkKind.FORM.value,
    "form_area": ChunkKind.FORM.value,
    "key_value_region": ChunkKind.KEY_VALUE.value,
    "key_value_area": ChunkKind.KEY_VALUE.value,
}


def _classify_kind(labels: list[str]) -> str:
    kinds = {_LABEL_TO_KIND.get(label, ChunkKind.UNKNOWN.value) for label in labels}
    if kinds == {ChunkKind.CAPTION.value, ChunkKind.PICTURE.value}:
        return ChunkKind.PICTURE.value
    if len(kinds) == 1:
        return next(iter(kinds))
    return ChunkKind.MIXED.value


def _picture_metadata(items: Iterable[Any]) -> tuple[list[dict[str, Any]], set[str]]:
    pictures: list[dict[str, Any]] = []
    with_descriptions: set[str] = set()
    for item in items:
        if not isinstance(item, PictureItem):
            continue
        description = None
        classification: list[dict[str, Any]] = []
        if item.meta is not None and item.meta.description is not None:
            description = {
                "label": DERIVED_VISUAL_DESCRIPTION_LABEL,
                "text": item.meta.description.text,
                "model_or_preset": item.meta.description.created_by,
                "generated_locally": True,
                "classification": "derived",
            }
            with_descriptions.add(item.self_ref)
        if item.meta is not None and item.meta.classification is not None:
            classification = [
                prediction.model_dump(mode="json")
                for prediction in item.meta.classification.predictions
            ]
        pictures.append(
            {
                "picture_ref": item.self_ref,
                "generated_description": description,
                "classification": classification,
            }
        )
    return pictures, with_descriptions


def _artifact_reference(artifact: Any | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return {
        "artifact_id": str(artifact.id),
        "artifact_type": artifact.artifact_type,
        "storage_path": artifact.storage_path,
        "page_no": artifact.page_no,
        "doc_item_ref": artifact.doc_item_ref,
    }


def _chunk_identity(
    processing_run_id: uuid.UUID,
    role: str,
    fingerprint: str,
    ordinal: int,
    raw_text: str,
    embedding_text: str,
    item_refs: list[str],
) -> uuid.UUID:
    content_hash = hashlib.sha256(
        (raw_text + "\0" + embedding_text).encode("utf-8")
    ).hexdigest()
    identity = _canonical_json(
        {
            "processing_run_id": str(processing_run_id),
            "chunk_role": role,
            "chunking_fingerprint": fingerprint,
            "ordinal": ordinal,
            "content_hash": content_hash,
            "doc_item_refs": sorted(set(item_refs)),
        }
    )
    return uuid.uuid5(CHUNK_ID_NAMESPACE, identity)


def _provenance_identity(chunk_id: uuid.UUID, region: ProvenanceRegion) -> uuid.UUID:
    return uuid.uuid5(
        PROVENANCE_ID_NAMESPACE,
        _canonical_json({"chunk_id": str(chunk_id), "region": region.stable_key}),
    )


def _build_draft(
    *,
    run: ProcessingRun,
    role: str,
    ordinal: int,
    chunk: BaseChunk,
    chunker: Any,
    tokenizer: TokenizerResolution,
    configuration: ChunkingConfiguration,
    configuration_payload: dict[str, Any],
    fingerprint: str,
    document: DoclingDocument,
    artifacts_by_ref: dict[tuple[str, str], Any],
    source_json_sha256: str,
    source_hierarchical_ordinals: list[int],
    split_detected: bool,
    table_chunk_counts: Counter[str],
) -> ChunkDraft:
    raw_text = chunk.text
    embedding_text = chunker.contextualize(chunk)
    raw_tokens = tokenizer.adapter.count_tokens(raw_text)
    contextualized_tokens = tokenizer.adapter.count_tokens(embedding_text)
    item_refs = list(dict.fromkeys(item.self_ref for item in chunk.meta.doc_items))
    items = [RefItem(cref=item_ref).resolve(document) for item_ref in item_refs]
    labels = [getattr(getattr(item, "label", None), "value", "unknown") for item in items]
    tables = [item.self_ref for item in items if isinstance(item, TableItem)]
    picture_components, derived_picture_refs = _picture_metadata(items)
    pictures = [entry["picture_ref"] for entry in picture_components]
    captions = _captions(items, document)
    classification = (
        ContentClassification.MIXED.value
        if derived_picture_refs and (captions or len(items) > len(derived_picture_refs))
        else ContentClassification.DERIVED.value
        if derived_picture_refs
        else ContentClassification.SOURCE.value
    )
    provenance = extract_provenance(
        document,
        item_refs,
        derived_picture_refs=derived_picture_refs,
    )
    pages = sorted({region.page_no for region in provenance.regions})
    table_artifacts = [
        reference
        for ref in tables
        if (reference := _artifact_reference(artifacts_by_ref.get((ArtifactType.TABLE_IMAGE.value, ref))))
    ]
    picture_artifacts = [
        reference
        for ref in pictures
        if (reference := _artifact_reference(artifacts_by_ref.get((ArtifactType.PICTURE_IMAGE.value, ref))))
    ]
    for component in picture_components:
        component["source_captions"] = [
            caption for caption in captions if caption["doc_item_ref"] in item_refs
        ]
        component["picture_artifact"] = next(
            (
                ref
                for ref in picture_artifacts
                if ref["doc_item_ref"] == component["picture_ref"]
            ),
            None,
        )
    header_status: str | None = None
    if tables:
        header_status = (
            HeaderRepetitionStatus.NOT_APPLICABLE.value
            if all(table_chunk_counts[ref] == 1 for ref in tables)
            else HeaderRepetitionStatus.NOT_REPEATED.value
        )
    overflow = role == ChunkRole.VECTOR_INDEX.value and contextualized_tokens > configuration.max_tokens
    source_item_texts = [
        {"doc_item_ref": item.self_ref, "label": item.label.value, "text": item.text}
        for item in items
        if isinstance(item, FormulaItem | CodeItem)
    ]
    chunk_id = _chunk_identity(
        run.id,
        role,
        fingerprint,
        ordinal,
        raw_text,
        embedding_text,
        item_refs,
    )
    return ChunkDraft(
        id=chunk_id,
        role=role,
        ordinal=ordinal,
        kind=_classify_kind(labels),
        raw_text=raw_text,
        embedding_text=embedding_text,
        raw_token_count=raw_tokens,
        contextualized_token_count=contextualized_tokens,
        section_path=list(chunk.meta.headings or []),
        captions=captions,
        doc_item_refs=item_refs,
        page_start=pages[0] if pages else None,
        page_end=pages[-1] if pages else None,
        table_ref=tables[0] if tables else None,
        picture_ref=pictures[0] if pictures else None,
        content_classification=classification,
        serializer_metadata=configuration_payload["serializers"],
        chunk_metadata={
            "configuration": configuration_payload,
            "docling_json_sha256": source_json_sha256,
            "item_labels": labels,
            "table_refs": tables,
            "picture_refs": pictures,
            "table_artifacts": table_artifacts,
            "picture_components": picture_components,
            "picture_artifacts": picture_artifacts,
            "source_item_texts": source_item_texts,
            "missing_provenance_item_refs": list(provenance.missing_item_refs),
            "source_hierarchical_ordinals": source_hierarchical_ordinals,
            "split_detected": split_detected,
            "merge_detected": len(source_hierarchical_ordinals) > 1,
            "fixed_token_overlap": None,
            "header_repetition_requested": configuration.repeat_table_header,
            "overflow_reason": (
                "unsplittable element exceeded the contextualized token limit"
                if overflow
                else None
            ),
        },
        header_repetition_status=header_status,
        overflow=overflow,
        provenance=provenance.regions,
    )


def _statistics(chunks: Iterable[ChunkDraft | Chunk]) -> ChunkStatistics:
    values = list(chunks)
    if not values:
        return ChunkStatistics(0, 0, 0.0, 0, 0, 0.0, 0, 0, {}, 0, 0, 0)
    raw = [chunk.raw_token_count for chunk in values]
    contextualized = [chunk.contextualized_token_count for chunk in values]
    return ChunkStatistics(
        count=len(values),
        raw_min=min(raw),
        raw_average=round(mean(raw), 2),
        raw_max=max(raw),
        contextualized_min=min(contextualized),
        contextualized_average=round(mean(contextualized), 2),
        contextualized_max=max(contextualized),
        overflow_count=sum(1 for chunk in values if chunk.overflow),
        counts_by_kind=dict(sorted(Counter(chunk.kind for chunk in values).items())),
        with_headings=sum(1 for chunk in values if chunk.section_path),
        with_captions=sum(1 for chunk in values if chunk.captions),
        missing_provenance=sum(
            1
            for chunk in values
            if bool(chunk.chunk_metadata.get("missing_provenance_item_refs"))
        ),
    )


def _result_from_existing(
    *,
    run_id: uuid.UUID,
    json_path: Path,
    json_sha256: str,
    configuration_payload: dict[str, Any],
    fingerprint: str,
    tokenizer: TokenizerResolution,
    chunks: list[Chunk],
    total_duration_ms: int,
) -> ChunkGenerationResult:
    hierarchical = [c for c in chunks if c.chunk_role == ChunkRole.HIERARCHICAL_INSPECTION.value]
    hybrid = [c for c in chunks if c.chunk_role == ChunkRole.VECTOR_INDEX.value]
    return ChunkGenerationResult(
        processing_run_id=run_id,
        docling_json_path=str(json_path),
        docling_json_sha256=json_sha256,
        configuration=configuration_payload,
        configuration_fingerprint=fingerprint,
        tokenizer_encoding=tokenizer.encoding_name,
        tokenizer_fallback_used=tokenizer.fallback_used,
        hierarchical=_statistics(hierarchical),
        hybrid=_statistics(hybrid),
        provenance_count=sum(len(chunk.provenance_records) for chunk in chunks),
        warnings=(),
        timings_ms={"total_backfill": total_duration_ms},
        created=False,
        reused=True,
        dry_run=False,
    )


def generate_chunks(
    processing_run_id: uuid.UUID,
    configuration: ChunkingConfiguration | None = None,
    *,
    force: bool = False,
    dry_run: bool = False,
    allow_running: bool = False,
    persist_progress: bool = False,
    settings: Settings | None = None,
) -> ChunkGenerationResult:
    """Generate or reuse a complete transactional chunk set for one run."""

    total_started = perf_counter()
    settings = settings or get_settings()
    configuration = configuration or ChunkingConfiguration.from_settings(settings)
    tokenizer = resolve_tokenizer(configuration)
    chunk_repository = ChunkRepository()
    artifact_repository = ArtifactRepository()

    with session_scope() as session:
        run = session.scalar(
            select(ProcessingRun)
            .where(ProcessingRun.id == processing_run_id)
            .with_for_update()
        )
        if run is None:
            raise ValueError(f"Processing run not found: {processing_run_id}")
        allowed_statuses = {ProcessingStatus.COMPLETED.value}
        if allow_running:
            allowed_statuses.add(ProcessingStatus.RUNNING.value)
        if run.status not in allowed_statuses:
            raise ValueError(
                f"Processing run is not eligible for chunking in status {run.status}: "
                f"{processing_run_id}"
            )
        if run.pipeline_type not in ENABLED_DOCLING_PIPELINES:
            raise NotImplementedError(
                f"Chunk generation is not enabled for pipeline {run.pipeline_type!r}"
            )
        json_artifact = artifact_repository.get_for_run_type(
            session, run.id, ArtifactType.DOCLING_JSON.value
        )
        if json_artifact is None:
            raise FileNotFoundError("Completed run has no registered lossless Docling JSON artifact")
        json_path = Path(json_artifact.storage_path).resolve()
        if not json_path.is_file():
            raise FileNotFoundError(f"Registered Docling JSON artifact is missing: {json_path}")
        json_sha256 = sha256_file(json_path)
        configuration_payload = build_configuration_payload(
            run.pipeline_type, configuration, tokenizer
        )
        fingerprint = _fingerprint(configuration_payload)
        existing = chunk_repository.list_for_run(session, run.id)
        matching = [chunk for chunk in existing if chunk.chunking_fingerprint == fingerprint]
        matching_source = bool(matching) and all(
            chunk.chunk_metadata.get("docling_json_sha256") == json_sha256
            for chunk in matching
        )
        matching_roles = {chunk.chunk_role for chunk in matching}
        if (
            not force
            and matching_source
            and matching_roles
            == {ChunkRole.HIERARCHICAL_INSPECTION.value, ChunkRole.VECTOR_INDEX.value}
            and len(matching) == len(existing)
        ):
            return _result_from_existing(
                run_id=run.id,
                json_path=json_path,
                json_sha256=json_sha256,
                configuration_payload=configuration_payload,
                fingerprint=fingerprint,
                tokenizer=tokenizer,
                chunks=matching,
                total_duration_ms=round((perf_counter() - total_started) * 1000),
            )

        load_started = perf_counter()
        if persist_progress:
            run.stage = "hierarchical_chunking"
            run.progress_percent = max(run.progress_percent, 55)
            session.commit()
        try:
            document = DoclingDocument.load_from_json(json_path)
        except Exception as exc:
            raise ValueError(f"Registered Docling JSON artifact is corrupt: {json_path}") from exc
        load_duration = round((perf_counter() - load_started) * 1000)

        serializer_provider = Prompt7SerializerProvider()
        hierarchical_chunker = HierarchicalChunker(
            serializer_provider=serializer_provider,
            always_emit_headings=False,
            merge_list_items=True,
        )
        hierarchical_started = perf_counter()
        hierarchical_chunks = list(hierarchical_chunker.chunk(document))
        hierarchical_duration = round((perf_counter() - hierarchical_started) * 1000)

        if persist_progress:
            run.stage = "hybrid_chunking"
            run.progress_percent = max(run.progress_percent, 68)
            session.commit()
        hybrid_chunker = HybridChunker(
            tokenizer=tokenizer.adapter,
            repeat_table_header=configuration.repeat_table_header,
            merge_peers=configuration.merge_peers,
            omit_header_on_overflow=configuration.omit_header_on_overflow,
            serializer_provider=serializer_provider,
            always_emit_headings=False,
        )
        hybrid_started = perf_counter()
        hybrid_chunks = list(hybrid_chunker.chunk(document))
        hybrid_duration = round((perf_counter() - hybrid_started) * 1000)

        artifacts = artifact_repository.list_for_run(session, run.id)
        artifacts_by_ref = {
            (artifact.artifact_type, artifact.doc_item_ref): artifact
            for artifact in artifacts
            if artifact.doc_item_ref is not None
        }
        hierarchical_refs = [
            set(item.self_ref for item in chunk.meta.doc_items)
            for chunk in hierarchical_chunks
        ]
        hybrid_ref_counts: Counter[str] = Counter(
            item.self_ref for chunk in hybrid_chunks for item in chunk.meta.doc_items
        )
        table_refs = {table.self_ref for table in document.tables}
        table_chunk_counts: Counter[str] = Counter(
            item.self_ref
            for chunk in hybrid_chunks
            for item in chunk.meta.doc_items
            if item.self_ref in table_refs
        )

        provenance_started = perf_counter()
        drafts: list[ChunkDraft] = []
        for role, chunker, chunks in (
            (
                ChunkRole.HIERARCHICAL_INSPECTION.value,
                hierarchical_chunker,
                hierarchical_chunks,
            ),
            (ChunkRole.VECTOR_INDEX.value, hybrid_chunker, hybrid_chunks),
        ):
            for ordinal, chunk in enumerate(chunks):
                refs = {item.self_ref for item in chunk.meta.doc_items}
                source_ordinals = [
                    index for index, natural_refs in enumerate(hierarchical_refs) if refs & natural_refs
                ]
                draft = _build_draft(
                    run=run,
                    role=role,
                    ordinal=ordinal,
                    chunk=chunk,
                    chunker=chunker,
                    tokenizer=tokenizer,
                    configuration=configuration,
                    configuration_payload=configuration_payload,
                    fingerprint=fingerprint,
                    document=document,
                    artifacts_by_ref=artifacts_by_ref,
                    source_json_sha256=json_sha256,
                    source_hierarchical_ordinals=source_ordinals,
                    split_detected=(
                        role == ChunkRole.VECTOR_INDEX.value
                        and any(hybrid_ref_counts[ref] > 1 for ref in refs)
                    ),
                    table_chunk_counts=table_chunk_counts,
                )
                drafts.append(draft)
        provenance_duration = round((perf_counter() - provenance_started) * 1000)

        warnings: list[str] = []
        for draft in drafts:
            missing = draft.chunk_metadata["missing_provenance_item_refs"]
            if missing:
                warnings.append(
                    f"{draft.role} chunk {draft.ordinal} has items without provenance: {missing}"
                )
            if draft.overflow:
                warnings.append(
                    f"vector chunk {draft.ordinal} is an unavoidable overflow at "
                    f"{draft.contextualized_token_count} tokens"
                )
        if dry_run:
            hierarchical_drafts = [
                draft for draft in drafts if draft.role == ChunkRole.HIERARCHICAL_INSPECTION.value
            ]
            hybrid_drafts = [draft for draft in drafts if draft.role == ChunkRole.VECTOR_INDEX.value]
            return ChunkGenerationResult(
                processing_run_id=run.id,
                docling_json_path=str(json_path),
                docling_json_sha256=json_sha256,
                configuration=configuration_payload,
                configuration_fingerprint=fingerprint,
                tokenizer_encoding=tokenizer.encoding_name,
                tokenizer_fallback_used=tokenizer.fallback_used,
                hierarchical=_statistics(hierarchical_drafts),
                hybrid=_statistics(hybrid_drafts),
                provenance_count=sum(len(draft.provenance) for draft in drafts),
                warnings=tuple(warnings),
                timings_ms={
                    "docling_json_load": load_duration,
                    "hierarchical_chunking": hierarchical_duration,
                    "hybrid_chunking": hybrid_duration,
                    "provenance_extraction": provenance_duration,
                    "persistence": 0,
                    "total_backfill": round((perf_counter() - total_started) * 1000),
                },
                created=False,
                reused=False,
                dry_run=True,
            )

        persistence_started = perf_counter()
        if existing:
            session.execute(delete(Chunk).where(Chunk.processing_run_id == run.id))
            session.flush()
        entities: list[Chunk] = []
        provenance_entities: list[ProvenanceRecord] = []
        for draft in drafts:
            entity = Chunk(
                id=draft.id,
                document_id=run.document_id,
                processing_run_id=run.id,
                ordinal=draft.ordinal,
                chunk_role=draft.role,
                kind=draft.kind,
                raw_text=draft.raw_text,
                embedding_text=draft.embedding_text,
                token_count=draft.contextualized_token_count,
                raw_token_count=draft.raw_token_count,
                contextualized_token_count=draft.contextualized_token_count,
                max_token_count=configuration.max_tokens,
                section_path=draft.section_path,
                captions=draft.captions,
                doc_item_refs=draft.doc_item_refs,
                page_start=draft.page_start,
                page_end=draft.page_end,
                table_ref=draft.table_ref,
                picture_ref=draft.picture_ref,
                is_derived_content=draft.content_classification
                != ContentClassification.SOURCE.value,
                content_classification=draft.content_classification,
                chunking_fingerprint=fingerprint,
                serializer_metadata=draft.serializer_metadata,
                chunk_metadata=draft.chunk_metadata,
                header_repetition_status=draft.header_repetition_status,
                overflow=draft.overflow,
                vector_collection=None,
                vector_id=None,
            )
            entities.append(entity)
            provenance_entities.extend(
                ProvenanceRecord(
                    id=_provenance_identity(draft.id, region),
                    chunk_id=draft.id,
                    document_id=run.document_id,
                    processing_run_id=run.id,
                    doc_item_ref=region.doc_item_ref,
                    page_no=region.page_no,
                    bbox_left=region.bbox_left,
                    bbox_top=region.bbox_top,
                    bbox_right=region.bbox_right,
                    bbox_bottom=region.bbox_bottom,
                    coordinate_origin=region.coordinate_origin,
                    char_start=region.char_start,
                    char_end=region.char_end,
                    evidence_role=region.evidence_role,
                )
                for region in draft.provenance
            )
        session.add_all(entities)
        session.add_all(provenance_entities)
        session.flush()
        persistence_duration = round((perf_counter() - persistence_started) * 1000)
        total_duration = round((perf_counter() - total_started) * 1000)
        timings = {
            "docling_json_load": load_duration,
            "hierarchical_chunking": hierarchical_duration,
            "hybrid_chunking": hybrid_duration,
            "provenance_extraction": provenance_duration,
            "persistence": persistence_duration,
            "total_backfill": total_duration,
        }
        for entity in entities:
            entity.chunk_metadata = {**entity.chunk_metadata, "generation_timings_ms": timings}

        hierarchical_drafts = [
            draft for draft in drafts if draft.role == ChunkRole.HIERARCHICAL_INSPECTION.value
        ]
        hybrid_drafts = [draft for draft in drafts if draft.role == ChunkRole.VECTOR_INDEX.value]
        return ChunkGenerationResult(
            processing_run_id=run.id,
            docling_json_path=str(json_path),
            docling_json_sha256=json_sha256,
            configuration=configuration_payload,
            configuration_fingerprint=fingerprint,
            tokenizer_encoding=tokenizer.encoding_name,
            tokenizer_fallback_used=tokenizer.fallback_used,
            hierarchical=_statistics(hierarchical_drafts),
            hybrid=_statistics(hybrid_drafts),
            provenance_count=len(provenance_entities),
            warnings=tuple(warnings),
            timings_ms=timings,
            created=True,
            reused=False,
            dry_run=False,
        )
