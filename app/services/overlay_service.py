"""Deterministic page-image overlays grounded in persisted provenance."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from docling_core.types.doc import BoundingBox, CoordOrigin, DoclingDocument, Size
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session

from app.core.enums import ArtifactType
from app.db.models import Artifact
from app.db.repositories import ArtifactRepository, ChunkRepository
from app.services.file_storage import FileStorage
from app.services.api_view_service import resolve_artifact_content


OVERLAY_ID_NAMESPACE = uuid.UUID("7ac459e4-707e-5ca6-b4b8-f9fdc114696e")
OVERLAY_STYLE_VERSION = "prompt7-v1"


class ChunkNotFoundError(LookupError):
    """The requested chunk does not exist."""


class PreciseProvenanceUnavailableError(ValueError):
    """No stored bounding box can ground the requested page."""


class OverlayPageRequiredError(ValueError):
    """A multi-page chunk requires an explicit page choice."""

    def __init__(self, pages: list[int]) -> None:
        self.pages = pages
        super().__init__(f"page_no is required; available pages: {pages}")


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """Generated or cached evidence overlay metadata."""

    artifact_id: uuid.UUID
    chunk_id: uuid.UUID
    page_no: int
    overlay_path: str
    source_page_image_path: str
    rectangle_count: int
    overlay_fingerprint: str
    coordinate_conversion: dict[str, Any]
    duration_ms: int
    reused: bool


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _precise(record: Any) -> bool:
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


def generate_evidence_overlay(
    session: Session,
    chunk_id: uuid.UUID,
    *,
    page_no: int | None = None,
    storage: FileStorage | None = None,
) -> OverlayResult:
    """Draw all exact source regions for one chunk page and cache the PNG."""

    started = perf_counter()
    chunk_repository = ChunkRepository()
    artifact_repository = ArtifactRepository()
    chunk = chunk_repository.get_with_provenance(session, chunk_id)
    if chunk is None:
        raise ChunkNotFoundError(f"Chunk not found: {chunk_id}")
    precise_records = [record for record in chunk.provenance_records if _precise(record)]
    pages = sorted({record.page_no for record in precise_records})
    if not pages:
        raise PreciseProvenanceUnavailableError(
            "Chunk has no precise bounding-box provenance; no image was fabricated"
        )
    if page_no is None:
        if len(pages) != 1:
            raise OverlayPageRequiredError(pages)
        page_no = pages[0]
    page_records = [record for record in precise_records if record.page_no == page_no]
    if not page_records:
        raise PreciseProvenanceUnavailableError(
            f"Chunk has no precise bounding-box provenance on page {page_no}; available pages: {pages}"
        )

    source_artifact = artifact_repository.get_for_run_type(
        session,
        chunk.processing_run_id,
        ArtifactType.PAGE_IMAGE.value,
        page_no=page_no,
    )
    if source_artifact is None:
        raise FileNotFoundError(f"No registered page-image artifact for page {page_no}")
    source_path, _source_mime = resolve_artifact_content(source_artifact)
    json_artifact = artifact_repository.get_for_run_type(
        session, chunk.processing_run_id, ArtifactType.DOCLING_JSON.value
    )
    if json_artifact is None:
        raise FileNotFoundError("The authoritative Docling JSON artifact is unavailable")
    json_path, _json_mime = resolve_artifact_content(json_artifact)
    document = DoclingDocument.load_from_json(json_path)
    page = document.pages.get(page_no)
    if page is None:
        raise ValueError(f"DoclingDocument has no page dimensions for page {page_no}")

    with Image.open(source_path) as opened:
        base_image = opened.convert("RGBA")
    rendered_size = Size(width=float(base_image.width), height=float(base_image.height))
    source_size = page.size
    stable_regions = [
        {
            "provenance_record_id": str(record.id),
            "doc_item_ref": record.doc_item_ref,
            "bbox": [
                record.bbox_left,
                record.bbox_top,
                record.bbox_right,
                record.bbox_bottom,
            ],
            "coordinate_origin": record.coordinate_origin,
            "evidence_role": record.evidence_role,
        }
        for record in sorted(page_records, key=lambda value: str(value.id))
    ]
    fingerprint_input = {
        "style_version": OVERLAY_STYLE_VERSION,
        "chunk_id": str(chunk.id),
        "page_no": page_no,
        "source_page_artifact_id": str(source_artifact.id),
        "source_page_size": source_size.model_dump(mode="json"),
        "rendered_size": rendered_size.model_dump(mode="json"),
        "regions": stable_regions,
    }
    fingerprint = hashlib.sha256(
        _canonical_json(fingerprint_input).encode("utf-8")
    ).hexdigest()
    cached = artifact_repository.get_overlay_by_fingerprint(
        session, chunk.processing_run_id, fingerprint
    )
    if cached is not None and Path(cached.storage_path).is_file():
        return OverlayResult(
            artifact_id=cached.id,
            chunk_id=chunk.id,
            page_no=page_no,
            overlay_path=cached.storage_path,
            source_page_image_path=str(source_path),
            rectangle_count=int(cached.metadata_json["rectangle_count"]),
            overlay_fingerprint=fingerprint,
            coordinate_conversion=cached.metadata_json["coordinate_conversion"],
            duration_ms=round((perf_counter() - started) * 1000),
            reused=True,
        )

    colors = [
        (239, 68, 68, 230),
        (37, 99, 235, 230),
        (5, 150, 105, 230),
        (217, 119, 6, 230),
    ]
    fill_colors = [(red, green, blue, 42) for red, green, blue, _alpha in colors]
    drawing_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(drawing_layer)
    rendered_rectangles: list[dict[str, Any]] = []
    line_width = max(3, round(base_image.width / 300))
    for index, record in enumerate(page_records):
        source_bbox = BoundingBox(
            l=record.bbox_left,
            t=record.bbox_top,
            r=record.bbox_right,
            b=record.bbox_bottom,
            coord_origin=CoordOrigin(record.coordinate_origin),
        )
        top_left_bbox = source_bbox.to_top_left_origin(source_size.height)
        scaled = top_left_bbox.scale_to_size(source_size, rendered_size)
        left = max(0.0, min(float(base_image.width - 1), min(scaled.l, scaled.r)))
        right = max(0.0, min(float(base_image.width - 1), max(scaled.l, scaled.r)))
        top = max(0.0, min(float(base_image.height - 1), min(scaled.t, scaled.b)))
        bottom = max(0.0, min(float(base_image.height - 1), max(scaled.t, scaled.b)))
        color = colors[index % len(colors)]
        fill = fill_colors[index % len(fill_colors)]
        draw.rectangle((left, top, right, bottom), outline=color, fill=fill, width=line_width)
        rendered_rectangles.append(
            {
                "provenance_record_id": str(record.id),
                "source_bbox": [
                    record.bbox_left,
                    record.bbox_top,
                    record.bbox_right,
                    record.bbox_bottom,
                ],
                "rendered_bbox": [left, top, right, bottom],
                "source_origin": record.coordinate_origin,
            }
        )
    output_image = Image.alpha_composite(base_image, drawing_layer).convert("RGB")
    storage = storage or FileStorage()
    relative_path = (
        f"overlays/{chunk.id}/page-{page_no:03d}-{fingerprint[:16]}.png"
    )
    output_path = storage.write_png_artifact(
        document_id=chunk.document_id,
        processing_run_id=chunk.processing_run_id,
        relative_path=relative_path,
        image=output_image,
    )
    conversion = {
        "helper": "BoundingBox.to_top_left_origin + BoundingBox.scale_to_size",
        "source_page_size": source_size.model_dump(mode="json"),
        "rendered_image_size": rendered_size.model_dump(mode="json"),
        "clipped_to_image_bounds": True,
        "rectangles": rendered_rectangles,
    }
    artifact_id = uuid.uuid5(OVERLAY_ID_NAMESPACE, fingerprint)
    artifact = Artifact(
        id=artifact_id,
        processing_run_id=chunk.processing_run_id,
        artifact_type=ArtifactType.EVIDENCE_OVERLAY.value,
        storage_path=str(output_path),
        page_no=page_no,
        doc_item_ref=None,
        metadata_json={
            "chunk_id": str(chunk.id),
            "page_no": page_no,
            "source_page_image_artifact_id": str(source_artifact.id),
            "source_processing_run_id": str(chunk.processing_run_id),
            "provenance_record_ids": [str(record.id) for record in page_records],
            "rectangle_count": len(page_records),
            "coordinate_conversion": conversion,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "overlay_fingerprint": fingerprint,
            "style_version": OVERLAY_STYLE_VERSION,
        },
    )
    session.add(artifact)
    session.flush()
    return OverlayResult(
        artifact_id=artifact_id,
        chunk_id=chunk.id,
        page_no=page_no,
        overlay_path=str(output_path),
        source_page_image_path=str(source_path),
        rectangle_count=len(page_records),
        overlay_fingerprint=fingerprint,
        coordinate_conversion=conversion,
        duration_ms=round((perf_counter() - started) * 1000),
        reused=False,
    )
