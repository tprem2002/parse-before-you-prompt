"""Compare original pages, baseline extraction, and Docling structure."""

from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import ApiClientError, get_api_client
from components.branding import (
    badge,
    page_header,
    render_sidebar,
    require_comparison_selection,
    setup_page,
)
from components.chunk_card import render_chunk_detail
from components.comparison_panel import baseline_limitations, representation_header
from components.errors import render_api_error
from state import cached_artifact_bytes, cached_artifacts, cached_chunk, cached_evidence_image


setup_page()
render_sidebar()
page_header(section="2 · Parsing comparison")
selection = require_comparison_selection()
if selection is None:
    st.stop()
document_id, baseline_run_id, docling_run_id = selection
client = get_api_client()

try:
    document = client.get_document(document_id)
    page_count = int(document.get("page_count") or 1)
except ApiClientError as error:
    render_api_error(error, context="Document unavailable")
    st.stop()


def move_page(delta: int) -> None:
    st.session_state.selected_page = max(
        1, min(page_count, int(st.session_state.selected_page) + delta)
    )


def choose_page(page: int) -> None:
    st.session_state.selected_page = page


st.markdown("## Select a demonstration page")
shortcut_columns = st.columns(7)
shortcuts = [
    (2, "Reading order"),
    (4, "Table I"),
    (5, "Table II"),
    (6, "Chart"),
    (7, "Architecture"),
    (8, "Scanned OCR"),
    (9, "Formula + code"),
]
for column, (number, label) in zip(shortcut_columns, shortcuts, strict=True):
    with column:
        st.button(
            f"Page {number}\n{label}",
            key=f"comparison_shortcut_{number}",
            on_click=choose_page,
            args=(number,),
            width="stretch",
        )

control_columns = st.columns([1, 2, 1, 2, 2, 2])
control_columns[0].button(
    "← Previous",
    key="comparison_previous",
    disabled=st.session_state.selected_page <= 1,
    on_click=move_page,
    args=(-1,),
    width="stretch",
)
with control_columns[1]:
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        key="selected_page",
        step=1,
    )
control_columns[2].button(
    "Next →",
    key="comparison_next",
    disabled=st.session_state.selected_page >= page_count,
    on_click=move_page,
    args=(1,),
    width="stretch",
)
with control_columns[3]:
    kind_filter = st.selectbox(
        "Content kind",
        ["All", "text", "list", "table", "picture", "formula", "code", "mixed"],
        key="comparison_kind_filter",
    )
with control_columns[4]:
    show_metadata = st.checkbox(
        "Show source metadata",
        value=bool(st.session_state.ui_display_preferences.get("show_source_metadata", True)),
        key="comparison_show_metadata",
    )
with control_columns[5]:
    show_provenance = st.checkbox(
        "Show exact provenance",
        value=bool(st.session_state.ui_display_preferences.get("show_precise_provenance", True)),
        key="comparison_show_provenance",
    )

try:
    artifacts = cached_artifacts(client, docling_run_id)
    page_artifact = next(
        (
            item
            for item in artifacts
            if item.get("artifact_type") == "page_image" and item.get("page_no") == page
        ),
        None,
    )
    baseline_page = client.list_chunks(
        baseline_run_id, page=int(page), limit=20, include_text=True
    )
    docling_page = client.list_chunks(
        docling_run_id,
        chunk_role="hierarchical_inspection",
        kind=None if kind_filter == "All" else kind_filter,
        page=int(page),
        limit=50,
        include_text=False,
    )
except ApiClientError as error:
    render_api_error(error, context="Comparison data unavailable")
    st.stop()

docling_items = list(docling_page.get("items") or [])
selected_detail: dict[str, Any] | None = None
if docling_items:
    item_ids = [str(item["id"]) for item in docling_items]
    current = st.session_state.selected_hierarchical_chunk_id
    selected_id = st.selectbox(
        "Docling item",
        item_ids,
        index=item_ids.index(current) if current in item_ids else 0,
        format_func=lambda value: next(
            f"#{item['ordinal']} · {item['kind']} · page {item['page_start']}"
            for item in docling_items
            if str(item["id"]) == value
        ),
        key="comparison_docling_item",
    )
    st.session_state.selected_hierarchical_chunk_id = selected_id
    try:
        selected_detail = cached_chunk(client, selected_id)
    except ApiClientError as error:
        render_api_error(error, context="Structured item unavailable")

image_mode = st.radio(
    "Page image",
    ["Original page", "Evidence overlay"],
    horizontal=True,
    key="comparison_image_mode",
)


def find_matching_overlay(detail: dict[str, Any] | None) -> tuple[str, int] | None:
    if not detail:
        return None
    source_refs = {str(item) for item in detail.get("doc_item_refs") or []}
    try:
        candidates = client.list_chunks(
            docling_run_id,
            chunk_role="vector_index",
            page=int(page),
            limit=50,
            include_text=False,
        ).get("items") or []
        for candidate in candidates:
            if not candidate.get("overlay_available"):
                continue
            candidate_detail = cached_chunk(client, str(candidate["id"]))
            candidate_refs = {str(item) for item in candidate_detail.get("doc_item_refs") or []}
            if source_refs & candidate_refs:
                return str(candidate["id"]), int(page)
    except ApiClientError:
        return None
    return None


overlay_target = find_matching_overlay(selected_detail) if image_mode == "Evidence overlay" else None

original_column, baseline_column, docling_column = st.columns([1.02, 1, 1.15])
with original_column:
    representation_header("Original rendered page", "Registered Docling page-image artifact", accent="source")
    if image_mode == "Evidence overlay" and overlay_target:
        try:
            overlay = cached_evidence_image(client, *overlay_target)
            st.image(overlay, caption=f"Application-resolved evidence · page {page}", width="stretch")
        except ApiClientError as error:
            render_api_error(error, context="Evidence overlay unavailable")
    elif page_artifact:
        try:
            page_bytes = cached_artifact_bytes(client, str(page_artifact["id"]))
            st.image(page_bytes, caption=f"Original page render · page {page}", width="stretch")
            if image_mode == "Evidence overlay":
                st.info("No cached precise overlay maps to the selected item; the original page is shown.")
        except ApiClientError as error:
            render_api_error(error, context="Page image unavailable")
    else:
        st.warning("No registered page-image artifact is available for this page.")

with baseline_column:
    representation_header("Baseline extraction", "PyMuPDF + fixed-window chunks", accent="baseline")
    baseline_limitations()
    baseline_text = "\n\n".join(
        str(item.get("raw_text") or "") for item in baseline_page.get("items") or []
    ).strip()
    if baseline_text:
        st.text_area(
            "Page-relevant extracted text",
            value=baseline_text,
            height=620,
            disabled=True,
            key=f"baseline_page_text_{page}",
        )
        st.caption(
            "Fixed chunks may span multiple pages; the API returns every chunk whose broad page range includes this page."
        )
    else:
        st.warning("No baseline text chunk covers this page.")

with docling_column:
    representation_header(
        "Docling structured representation",
        "HierarchicalChunker inspection + exact item provenance",
        accent="docling",
    )
    if selected_detail:
        classification = str(selected_detail.get("content_classification") or "unknown")
        st.markdown(
            badge("Direct source evidence" if classification == "source" else classification, "cyan" if classification == "source" else "amber"),
            unsafe_allow_html=True,
        )
        render_chunk_detail(selected_detail, show_embedding_text=False)
        if show_metadata:
            st.write(
                {
                    "item_kind": selected_detail.get("kind"),
                    "page": selected_detail.get("page_start"),
                    "doc_item_refs": selected_detail.get("doc_item_refs"),
                    "table_ref": selected_detail.get("table_ref"),
                    "picture_ref": selected_detail.get("picture_ref"),
                    "provenance_available": selected_detail.get("precise_provenance_available"),
                }
            )
        if show_provenance:
            provenance = selected_detail.get("provenance") or []
            st.caption(f"Exact stored source regions: {len(provenance)}")
            for region in provenance:
                st.text(
                    f"{region.get('doc_item_ref')} · page {region.get('page_no')} · "
                    f"{region.get('coordinate_origin')} · {region.get('evidence_role')}"
                )
    else:
        st.info("No Docling items match this page and content-kind filter.")

if int(page) == 8:
    docling_text = "\n".join(
        str(item.get("raw_text") or "")
        for item in client.list_chunks(
            docling_run_id,
            chunk_role="hierarchical_inspection",
            page=8,
            limit=50,
            include_text=True,
        ).get("items")
        or []
    )
    recovered_lines = [
        line.strip() for line in docling_text.splitlines() if "recovery window" in line.lower()
    ]
    baseline_has_same_content = any(
        line.lower() in baseline_text.lower() for line in recovered_lines if line
    )
    if recovered_lines and not baseline_has_same_content:
        st.success(
            "Measured page-8 contrast: the baseline representation does not contain the scanned "
            "recovery-window line, while Docling OCR preserved:\n\n"
            + "\n".join(recovered_lines)
        )
