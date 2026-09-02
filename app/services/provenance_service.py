"""Exact Docling item-to-source provenance extraction."""

from __future__ import annotations

from dataclasses import dataclass

from docling_core.types.doc import DoclingDocument, PictureItem, RefItem, TableItem

from app.core.enums import ProvenanceRole


@dataclass(frozen=True, slots=True)
class ProvenanceRegion:
    """One real source region copied from a Docling ProvenanceItem."""

    doc_item_ref: str
    page_no: int
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float
    coordinate_origin: str
    char_start: int | None
    char_end: int | None
    evidence_role: str

    @property
    def stable_key(self) -> tuple[object, ...]:
        """Return the complete de-duplication key for this region."""

        return (
            self.doc_item_ref,
            self.page_no,
            self.bbox_left,
            self.bbox_top,
            self.bbox_right,
            self.bbox_bottom,
            self.coordinate_origin,
            self.char_start,
            self.char_end,
            self.evidence_role,
        )


@dataclass(frozen=True, slots=True)
class ProvenanceExtraction:
    """Resolved regions and item references that had no usable provenance."""

    regions: tuple[ProvenanceRegion, ...]
    missing_item_refs: tuple[str, ...]


def extract_provenance(
    document: DoclingDocument,
    doc_item_refs: list[str],
    *,
    derived_picture_refs: set[str] | None = None,
) -> ProvenanceExtraction:
    """Resolve every item reference and retain every supplied source region.

    Coordinates are copied exactly as stored by Docling. No text search, page
    inference, coordinate normalization, or approximation happens here.
    """

    derived_picture_refs = derived_picture_refs or set()
    regions_by_key: dict[tuple[object, ...], ProvenanceRegion] = {}
    missing: list[str] = []
    for doc_item_ref in dict.fromkeys(doc_item_refs):
        try:
            item = RefItem(cref=doc_item_ref).resolve(document)
        except (IndexError, KeyError, ValueError):
            missing.append(doc_item_ref)
            continue
        provenance = getattr(item, "prov", None) or []
        if not provenance:
            missing.append(doc_item_ref)
            continue
        if isinstance(item, PictureItem):
            role = (
                ProvenanceRole.DERIVED_VISUAL_ANCHOR.value
                if doc_item_ref in derived_picture_refs
                else ProvenanceRole.SOURCE_IMAGE_REGION.value
            )
        elif isinstance(item, TableItem):
            role = ProvenanceRole.SOURCE_IMAGE_REGION.value
        else:
            role = ProvenanceRole.DIRECT_SOURCE_TEXT.value
        for source in provenance:
            charspan = source.charspan if source.charspan is not None else (None, None)
            region = ProvenanceRegion(
                doc_item_ref=doc_item_ref,
                page_no=source.page_no,
                bbox_left=source.bbox.l,
                bbox_top=source.bbox.t,
                bbox_right=source.bbox.r,
                bbox_bottom=source.bbox.b,
                coordinate_origin=source.bbox.coord_origin.value,
                char_start=charspan[0],
                char_end=charspan[1],
                evidence_role=role,
            )
            regions_by_key.setdefault(region.stable_key, region)
    return ProvenanceExtraction(
        regions=tuple(regions_by_key.values()),
        missing_item_refs=tuple(dict.fromkeys(missing)),
    )
