"""Local Docling 2.123.1 standard conversion and artifact export."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import threading
import uuid
import warnings as python_warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.core.config import Settings, get_settings
from app.core.enums import ArtifactType, PipelineType
from app.core.logging import get_logger
from app.services.file_storage import FileStorage


if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument, PictureItem, TableItem


logger = get_logger(__name__)

DERIVED_VISUAL_DESCRIPTION_LABEL = (
    "Derived visual description — generated locally from the source image"
)
PICTURE_DESCRIPTION_PRESET = "granite_vision"
PICTURE_DESCRIPTION_MODEL = "ibm-granite/granite-vision-3.3-2b"
PICTURE_CLASSIFICATION_PRESET = "document_figure_classifier_v2"
LAYOUT_PRESET = "layout_heron_default"
CODE_FORMULA_PRESET = "codeformulav2"
OCR_ENGINE = "rapidocr"
OCR_BACKEND = "onnxruntime"
THREADED_PARSER_THREADS = 1


@dataclass(frozen=True, slots=True)
class ConverterHandle:
    """A singleton converter paired with the configuration that created it."""

    converter: DocumentConverter
    configuration_fingerprint: str
    configuration: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StandardConversion:
    """One local conversion result plus warnings and measured runtime."""

    result: ConversionResult
    converter_fingerprint: str
    warnings: tuple[dict[str, Any], ...]
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Filesystem artifact information ready for PostgreSQL persistence."""

    artifact_type: str
    storage_path: str
    page_no: int | None = None
    doc_item_ref: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DoclingExport:
    """All artifact descriptors and the manifest generated for a conversion."""

    artifacts: tuple[ArtifactDescriptor, ...]
    manifest: dict[str, Any]


_converter_lock = threading.RLock()
_converter_handle: ConverterHandle | None = None


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def docling_dependency_versions() -> dict[str, str]:
    """Return the exact local packages that materially affect conversion."""

    return {
        name: _package_version(name)
        for name in (
            "docling",
            "docling-core",
            "docling-ibm-models",
            "docling-parse",
            "transformers",
            "torch",
            "rapidocr",
            "onnxruntime",
        )
    }


def _canonical_fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_artifacts_path(settings: Settings) -> str | None:
    if settings.docling_artifacts_path is None:
        return None
    return str(settings.docling_artifacts_path.resolve())


def converter_configuration(settings: Settings | None = None) -> dict[str, Any]:
    """Describe every setting that changes the process-level converter."""

    settings = settings or get_settings()
    return {
        "pipeline_options_class": "ThreadedPdfPipelineOptions",
        "pdf_backend": "ThreadedDoclingParseDocumentBackend",
        "pdf_backend_options": {
            "options_class": "ThreadedDoclingParseBackendOptions",
            "parser_threads": THREADED_PARSER_THREADS,
            "rationale": (
                "Serialize native docling-parse decoding on Windows; model stages still use "
                "the configured accelerator thread count"
            ),
            "release_native_memory_every_n_pages": 128,
        },
        "enable_remote_services": False,
        "allow_external_plugins": False,
        "artifacts_path": _resolved_artifacts_path(settings),
        "accelerator": {
            "device": settings.docling_device,
            "num_threads": settings.docling_num_threads,
        },
        "limits": {
            "max_file_size_mb": settings.docling_max_file_size_mb,
            "max_file_size_bytes": settings.docling_max_file_size_mb * 1024 * 1024,
            "max_pages": settings.docling_max_pages,
        },
        "ocr": {
            "enabled": True,
            "options_class": "RapidOcrOptions",
            "engine": OCR_ENGINE,
            "backend": OCR_BACKEND,
            "languages": ["english"],
            "mode": "default",
            "scale": 3.0,
        },
        "tables": {
            "enabled": True,
            "options_class": "TableStructureOptions",
            "mode": "accurate",
            "cell_matching": True,
            "generate_table_images": False,
            "table_image_extraction": "TableItem.get_image(document) from retained page images",
        },
        "layout": {
            "options_class": "LayoutObjectDetectionOptions",
            "preset": LAYOUT_PRESET,
            "model": "docling-project/docling-layout-heron",
        },
        "heading_hierarchy": {
            "enabled": True,
            "options_class": "HeadingHierarchyOptions",
            "use_bookmarks": True,
            "use_numbering": True,
            "use_style": True,
            "use_font_style": True,
            "max_level": 6,
            "generate_parsed_pages": True,
        },
        "images": {
            "scale": settings.docling_image_scale,
            "generate_page_images": True,
            "generate_picture_images": True,
        },
        "picture_classification": {
            "enabled": True,
            "options_class": "DocumentPictureClassifierOptions",
            "preset": PICTURE_CLASSIFICATION_PRESET,
            "model": "docling-project/DocumentFigureClassifier-v2.5",
        },
        "picture_description": {
            "enabled": settings.docling_enable_picture_description,
            "options_class": "PictureDescriptionVlmEngineOptions",
            "preset": PICTURE_DESCRIPTION_PRESET,
            "model": PICTURE_DESCRIPTION_MODEL,
            "engine": "auto_inline",
            "remote_api": False,
            "derived_content_label": DERIVED_VISUAL_DESCRIPTION_LABEL,
        },
        "code_formula": {
            "options_class": "CodeFormulaVlmOptions",
            "preset": CODE_FORMULA_PRESET,
            "model": "docling-project/CodeFormulaV2",
            "code_enabled": settings.docling_enable_code_enrichment,
            "formula_enabled": settings.docling_enable_formula_enrichment,
            "engine": "auto_inline",
            "remote_api": False,
        },
        "chart_extraction": {
            "enabled": settings.docling_enable_chart_extraction,
            "default": False,
        },
        "local_cache": {"hf_home": str(settings.hf_home.resolve())},
    }


def docling_standard_configuration(
    source_sha256: str, settings: Settings | None = None
) -> dict[str, Any]:
    """Return the complete immutable configuration used for run reuse."""

    settings = settings or get_settings()
    converter_config = converter_configuration(settings)
    configuration: dict[str, Any] = {
        "configuration_version": 1,
        "pipeline_type": PipelineType.DOCLING_STANDARD.value,
        "source_sha256": source_sha256,
        "dependencies": docling_dependency_versions(),
        "converter_configuration": converter_config,
        "converter_configuration_fingerprint": _canonical_fingerprint(converter_config),
    }
    configuration["configuration_fingerprint"] = _canonical_fingerprint(configuration)
    return configuration


def _configure_local_cache(settings: Settings) -> None:
    cache = settings.hf_home.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)


def _build_converter(settings: Settings, configuration: dict[str, Any]) -> DocumentConverter:
    """Construct the inspected Docling 2.123.1 standard PDF converter."""

    _configure_local_cache(settings)
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
    from docling.datamodel.pipeline_options import (
        CodeFormulaVlmOptions,
        DocumentPictureClassifierOptions,
        HeadingHierarchyOptions,
        LayoutObjectDetectionOptions,
        PictureDescriptionVlmEngineOptions,
        RapidOcrOptions,
        TableFormerMode,
        TableStructureOptions,
        ThreadedPdfPipelineOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
        ThreadedDoclingParseDocumentBackend,
    )

    picture_description = PictureDescriptionVlmEngineOptions.from_preset(
        PICTURE_DESCRIPTION_PRESET
    ).model_copy(update={"scale": settings.docling_image_scale})
    code_formula = CodeFormulaVlmOptions.from_preset(CODE_FORMULA_PRESET).model_copy(
        update={
            "scale": settings.docling_image_scale,
            "extract_code": settings.docling_enable_code_enrichment,
            "extract_formulas": settings.docling_enable_formula_enrichment,
        }
    )
    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=settings.docling_device,
            num_threads=settings.docling_num_threads,
        ),
        enable_remote_services=False,
        allow_external_plugins=False,
        artifacts_path=settings.docling_artifacts_path,
        do_ocr=True,
        ocr_options=RapidOcrOptions(lang=["english"], backend=OCR_BACKEND),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        ),
        layout_options=LayoutObjectDetectionOptions.from_preset(LAYOUT_PRESET),
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        generate_parsed_pages=True,
        images_scale=settings.docling_image_scale,
        generate_page_images=True,
        generate_picture_images=True,
        generate_table_images=False,
        do_picture_classification=True,
        picture_classification_options=DocumentPictureClassifierOptions.from_preset(
            PICTURE_CLASSIFICATION_PRESET
        ),
        do_picture_description=settings.docling_enable_picture_description,
        picture_description_options=picture_description,
        do_code_enrichment=settings.docling_enable_code_enrichment,
        do_formula_enrichment=settings.docling_enable_formula_enrichment,
        code_formula_options=code_formula,
        do_chart_extraction=settings.docling_enable_chart_extraction,
    )
    logger.info(
        "Initializing local Docling converter fingerprint=%s device=%s threads=%d",
        configuration["converter_configuration_fingerprint"],
        settings.docling_device,
        settings.docling_num_threads,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=ThreadedDoclingParseDocumentBackend,
                backend_options=ThreadedDoclingParseBackendOptions(
                    enable_remote_fetch=False,
                    enable_local_fetch=False,
                    parser_threads=THREADED_PARSER_THREADS,
                    release_native_memory_every_n_pages=128,
                ),
            )
        },
    )


def get_standard_converter(settings: Settings | None = None) -> ConverterHandle:
    """Lazily create or explicitly reconstruct the process-level converter."""

    global _converter_handle
    settings = settings or get_settings()
    configuration = docling_standard_configuration("converter-only", settings)
    fingerprint = configuration["converter_configuration_fingerprint"]
    with _converter_lock:
        if _converter_handle is not None:
            if _converter_handle.configuration_fingerprint == fingerprint:
                return _converter_handle
            logger.warning(
                "Reconstructing Docling converter after configuration changed from %s to %s",
                _converter_handle.configuration_fingerprint,
                fingerprint,
            )
        converter = _build_converter(settings, configuration)
        _converter_handle = ConverterHandle(
            converter=converter,
            configuration_fingerprint=fingerprint,
            configuration=configuration["converter_configuration"],
        )
        return _converter_handle


def convert_docling_standard(
    source: Path, settings: Settings | None = None
) -> StandardConversion:
    """Convert one local PDF without any baseline or remote-service fallback."""

    settings = settings or get_settings()
    source = source.resolve()
    maximum_bytes = settings.docling_max_file_size_mb * 1024 * 1024
    if source.stat().st_size > maximum_bytes:
        raise ValueError(
            f"PDF exceeds the configured {settings.docling_max_file_size_mb} MB Docling limit"
        )
    handle = get_standard_converter(settings)
    started = perf_counter()
    captured: list[dict[str, Any]] = []
    logged_warnings: list[dict[str, Any]] = []

    class _WarningCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logged_warnings.append(
                {
                    "source": "logging_warning",
                    "logger": record.name,
                    "message": record.getMessage(),
                }
            )

    warning_handler = _WarningCapture(level=logging.WARNING)
    root_logger = logging.getLogger()
    root_logger.addHandler(warning_handler)
    try:
        with python_warnings.catch_warnings(record=True) as emitted:
            python_warnings.simplefilter("always")
            result = handle.converter.convert(
                source,
                raises_on_error=False,
                max_num_pages=settings.docling_max_pages,
                max_file_size=maximum_bytes,
            )
    finally:
        root_logger.removeHandler(warning_handler)
    captured.extend(
        {
            "source": "python_warning",
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in emitted
    )
    captured.extend(logged_warnings)
    captured.extend(
        {"source": "docling_conversion_error", "detail": error.model_dump(mode="json")}
        for error in result.errors
    )
    return StandardConversion(
        result=result,
        converter_fingerprint=handle.configuration_fingerprint,
        warnings=tuple(captured),
        duration_ms=round((perf_counter() - started) * 1000),
    )


def _page_no(item: Any) -> int | None:
    provenance = getattr(item, "prov", None) or []
    return provenance[0].page_no if provenance else None


def _caption_texts(item: Any, document: DoclingDocument) -> list[str]:
    captions: list[str] = []
    for reference in getattr(item, "captions", None) or []:
        resolved = reference.resolve(document)
        text = getattr(resolved, "text", None)
        if text:
            captions.append(text)
    return captions


def _picture_metadata(picture: PictureItem, document: DoclingDocument) -> dict[str, Any]:
    classification: list[dict[str, Any]] = []
    description: dict[str, Any] | None = None
    meta = picture.meta
    if meta is not None and meta.classification is not None:
        classification = [
            prediction.model_dump(mode="json")
            for prediction in meta.classification.predictions
        ]
    if meta is not None and meta.description is not None:
        description = {
            "label": DERIVED_VISUAL_DESCRIPTION_LABEL,
            "text": meta.description.text,
            "model_or_preset": meta.description.created_by or PICTURE_DESCRIPTION_PRESET,
            "preset": PICTURE_DESCRIPTION_PRESET,
            "model": PICTURE_DESCRIPTION_MODEL,
            "generated_locally": True,
            "is_derived_content": True,
        }
    return {
        "picture_ref": picture.self_ref,
        "page_no": _page_no(picture),
        "classification": classification,
        "source_captions": _caption_texts(picture, document),
        "generated_description": description,
        "source_caption_overwritten": False,
    }


def _table_metadata(table: TableItem, document: DoclingDocument) -> dict[str, Any]:
    return {
        "table_ref": table.self_ref,
        "page_no": _page_no(table),
        "num_rows": table.data.num_rows,
        "num_cols": table.data.num_cols,
        "source_captions": _caption_texts(table, document),
        "cell_matching": True,
        "image_extraction": "TableItem.get_image(document) from retained page images",
    }


def _document_observations(document: DoclingDocument) -> dict[str, Any]:
    from docling_core.types.doc import CodeItem, FormulaItem, SectionHeaderItem

    counts: Counter[str] = Counter()
    headings: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    code: list[dict[str, Any]] = []
    for item, _level in document.iterate_items(with_groups=True, traverse_pictures=True):
        label = getattr(getattr(item, "label", None), "value", None)
        counts[label or type(item).__name__] += 1
        if isinstance(item, SectionHeaderItem):
            headings.append(
                {
                    "text": item.text,
                    "level": item.level,
                    "page_no": _page_no(item),
                    "doc_item_ref": item.self_ref,
                    "parent_ref": getattr(item.parent, "cref", None),
                }
            )
        elif isinstance(item, FormulaItem):
            formulas.append(
                {"text": item.text, "page_no": _page_no(item), "doc_item_ref": item.self_ref}
            )
        elif isinstance(item, CodeItem):
            code.append(
                {"text": item.text, "page_no": _page_no(item), "doc_item_ref": item.self_ref}
            )
    return {
        "element_counts_by_type": dict(sorted(counts.items())),
        "heading_hierarchy": headings,
        "formulas": formulas,
        "code": code,
    }


def _artifact_manifest_entry(artifact: ArtifactDescriptor) -> dict[str, Any]:
    return {
        "artifact_type": artifact.artifact_type,
        "storage_path": artifact.storage_path,
        "page_no": artifact.page_no,
        "doc_item_ref": artifact.doc_item_ref,
    }


def _resolved_runtime_device(requested_device: str) -> str:
    """Resolve Docling's actual accelerator choice for audit metadata."""

    from docling.utils.accelerator_utils import decide_device

    return decide_device(requested_device)


def export_docling_artifacts(
    *,
    conversion: StandardConversion,
    storage: FileStorage,
    document_id: uuid.UUID,
    processing_run_id: uuid.UUID,
    source_filename: str,
    source_sha256: str,
    source_path: Path,
    configuration: dict[str, Any],
    started_at: datetime,
    completed_at: datetime,
    duration_ms: int,
) -> DoclingExport:
    """Persist lossless, human-readable, image, and manifest outputs."""

    from docling.datamodel.base_models import ConversionStatus

    result = conversion.result
    status_value = result.status.value
    artifacts: list[ArtifactDescriptor] = [
        ArtifactDescriptor(
            artifact_type=ArtifactType.ORIGINAL_PDF.value,
            storage_path=str(source_path.resolve()),
            metadata={"filename": source_filename, "sha256": source_sha256},
        )
    ]
    observations: dict[str, Any] = {
        "element_counts_by_type": {},
        "heading_hierarchy": [],
        "formulas": [],
        "code": [],
    }
    picture_findings: list[dict[str, Any]] = []
    table_findings: list[dict[str, Any]] = []

    if result.status in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
        document = result.document
        lossless = document.export_to_dict(mode="json", by_alias=True, exclude_none=True)
        json_text = json.dumps(lossless, ensure_ascii=False, indent=2)
        json.loads(json_text)
        json_path = storage.write_artifact_text(
            document_id=document_id,
            processing_run_id=processing_run_id,
            relative_path="docling.json",
            text=json_text,
        )
        artifacts.append(
            ArtifactDescriptor(
                artifact_type=ArtifactType.DOCLING_JSON.value,
                storage_path=str(json_path),
                metadata={
                    "lossless": True,
                    "valid_json": True,
                    "schema": lossless.get("schema_name"),
                },
            )
        )
        markdown = document.export_to_markdown(
            include_annotations=False,
            traverse_pictures=True,
        )
        markdown_path = storage.write_artifact_text(
            document_id=document_id,
            processing_run_id=processing_run_id,
            relative_path="document.md",
            text=markdown,
        )
        artifacts.append(
            ArtifactDescriptor(
                artifact_type=ArtifactType.MARKDOWN.value,
                storage_path=str(markdown_path),
                metadata={
                    "encoding": "utf-8",
                    "annotations_included": False,
                    "derived_descriptions_are_in_manifest": True,
                },
            )
        )

        for page_key, page in sorted(document.pages.items()):
            if page.image is None:
                logger.warning("Page %s has no retained image", page.page_no)
                continue
            image = page.image.pil_image
            path = storage.write_png_artifact(
                document_id=document_id,
                processing_run_id=processing_run_id,
                relative_path=f"pages/page-{page.page_no:03d}.png",
                image=image,
            )
            artifacts.append(
                ArtifactDescriptor(
                    artifact_type=ArtifactType.PAGE_IMAGE.value,
                    storage_path=str(path),
                    page_no=page.page_no,
                    metadata={
                        "page_key": page_key,
                        "width_px": image.width,
                        "height_px": image.height,
                        "image_scale": configuration["converter_configuration"]["images"]["scale"],
                    },
                )
            )

        for ordinal, picture in enumerate(document.pictures, start=1):
            metadata = _picture_metadata(picture, document)
            picture_findings.append(metadata)
            image = picture.get_image(document)
            if image is None:
                logger.warning("Picture %s has no extractable image", picture.self_ref)
                continue
            path = storage.write_png_artifact(
                document_id=document_id,
                processing_run_id=processing_run_id,
                relative_path=f"pictures/picture-{ordinal:03d}.png",
                image=image,
            )
            artifacts.append(
                ArtifactDescriptor(
                    artifact_type=ArtifactType.PICTURE_IMAGE.value,
                    storage_path=str(path),
                    page_no=metadata["page_no"],
                    doc_item_ref=picture.self_ref,
                    metadata={**metadata, "width_px": image.width, "height_px": image.height},
                )
            )

        for ordinal, table in enumerate(document.tables, start=1):
            metadata = _table_metadata(table, document)
            table_findings.append(metadata)
            image = table.get_image(document)
            if image is None:
                logger.warning("Table %s has no extractable image", table.self_ref)
                continue
            path = storage.write_png_artifact(
                document_id=document_id,
                processing_run_id=processing_run_id,
                relative_path=f"tables/table-{ordinal:03d}.png",
                image=image,
            )
            artifacts.append(
                ArtifactDescriptor(
                    artifact_type=ArtifactType.TABLE_IMAGE.value,
                    storage_path=str(path),
                    page_no=metadata["page_no"],
                    doc_item_ref=table.self_ref,
                    metadata={**metadata, "width_px": image.width, "height_px": image.height},
                )
            )
        observations = _document_observations(document)

    manifest_path = storage.artifact_directory(
        document_id=document_id, processing_run_id=processing_run_id
    ) / "conversion-manifest.json"
    manifest_preview = ArtifactDescriptor(
        artifact_type=ArtifactType.CONVERSION_MANIFEST.value,
        storage_path=str(manifest_path),
        metadata={"manifest_schema_version": 1, "conversion_status": status_value},
    )
    manifest_artifacts = [*artifacts, manifest_preview]
    manifest = {
        "manifest_schema_version": 1,
        "document_id": str(document_id),
        "processing_run_id": str(processing_run_id),
        "source_filename": source_filename,
        "source_sha256": source_sha256,
        "docling_version": configuration["dependencies"]["docling"],
        "dependency_versions": configuration["dependencies"],
        "pipeline_type": PipelineType.DOCLING_STANDARD.value,
        "conversion_status": status_value,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_ms": duration_ms,
        "conversion_call_duration_ms": conversion.duration_ms,
        "device": _resolved_runtime_device(
            configuration["converter_configuration"]["accelerator"]["device"]
        ),
        "requested_device": configuration["converter_configuration"]["accelerator"]["device"],
        "cpu_thread_count": configuration["converter_configuration"]["accelerator"]["num_threads"],
        "configuration_fingerprint": configuration["configuration_fingerprint"],
        "converter_configuration_fingerprint": conversion.converter_fingerprint,
        "exact_pipeline_settings": configuration["converter_configuration"],
        "model_and_preset_identifiers": {
            "layout": LAYOUT_PRESET,
            "picture_classifier": PICTURE_CLASSIFICATION_PRESET,
            "picture_description": PICTURE_DESCRIPTION_PRESET,
            "code_formula": CODE_FORMULA_PRESET,
        },
        "page_count": len(result.document.pages),
        **observations,
        "table_count": len(result.document.tables),
        "picture_count": len(result.document.pictures),
        "formula_count": len(observations["formulas"]),
        "code_count": len(observations["code"]),
        "ocr_observations": {
            "enabled": True,
            "engine": OCR_ENGINE,
            "backend": OCR_BACKEND,
            "languages": ["english"],
            "parsed_pages_retained": all(page.parsed_page is not None for page in result.pages),
        },
        "warnings": list(conversion.warnings),
        "pictures": picture_findings,
        "tables": table_findings,
        "chart_extraction_enabled": configuration["converter_configuration"]["chart_extraction"]["enabled"],
        "artifacts": [_artifact_manifest_entry(artifact) for artifact in manifest_artifacts],
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2)
    json.loads(manifest_text)
    written_manifest_path = storage.write_artifact_text(
        document_id=document_id,
        processing_run_id=processing_run_id,
        relative_path="conversion-manifest.json",
        text=manifest_text,
    )
    artifacts.append(
        ArtifactDescriptor(
            artifact_type=ArtifactType.CONVERSION_MANIFEST.value,
            storage_path=str(written_manifest_path),
            metadata=manifest_preview.metadata,
        )
    )
    return DoclingExport(artifacts=tuple(artifacts), manifest=manifest)
