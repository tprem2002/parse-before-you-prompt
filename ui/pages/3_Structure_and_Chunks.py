"""Inspect Docling's recovered hierarchy and Hybrid embedding chunks."""

from __future__ import annotations

from collections import defaultdict
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
from components.chunk_card import page_range, render_chunk_detail
from components.errors import render_api_error
from state import cached_artifact_bytes, cached_artifacts, cached_chunk


setup_page()
render_sidebar()
page_header(section="3 · Structure and chunks")
selection = require_comparison_selection()
if selection is None:
    st.stop()
_document_id, _baseline_run_id, docling_run_id = selection
client = get_api_client()

hierarchy_tab, hybrid_tab = st.tabs(["Hierarchical structure", "Hybrid embedding chunks"])

with hierarchy_tab:
    st.markdown("## Recovered document hierarchy")
    st.info(
        "The structure view reflects Docling’s actual recovered hierarchy. Some visually "
        "inferred levels may be imperfect. The UI does not repair or rewrite heading levels."
    )
    hierarchy_page = st.selectbox(
        "Inspect page",
        list(range(1, 11)),
        index=max(0, min(9, int(st.session_state.selected_page) - 1)),
        key="hierarchy_page_filter",
    )
    try:
        hierarchy_response = client.list_chunks(
            docling_run_id,
            chunk_role="hierarchical_inspection",
            page=int(hierarchy_page),
            limit=100,
            include_text=False,
        )
        hierarchy_summaries = list(hierarchy_response.get("items") or [])
        hierarchy_details = [
            cached_chunk(client, str(item["id"])) for item in hierarchy_summaries
        ]
    except ApiClientError as error:
        render_api_error(error, context="Hierarchy unavailable")
        hierarchy_details = []

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for detail in hierarchy_details:
        path = tuple(str(item) for item in detail.get("section_path") or ["Document root"])
        grouped[path].append(detail)

    if grouped:
        st.caption(
            f"Page {hierarchy_page}: {len(hierarchy_details)} natural chunks grouped by their stored heading paths."
        )
        for path, chunks in grouped.items():
            with st.expander(" → ".join(path), expanded=len(grouped) == 1):
                for chunk in chunks:
                    st.markdown(
                        badge(f"#{chunk.get('ordinal')}", "violet")
                        + badge(str(chunk.get("kind")), "cyan")
                        + badge(f"{chunk.get('raw_token_count')} natural tokens", "cyan"),
                        unsafe_allow_html=True,
                    )
                    caption_text = "; ".join(str(item) for item in chunk.get("captions") or [])
                    st.caption(
                        f"{page_range(chunk)} · {len(chunk.get('doc_item_refs') or [])} item refs · "
                        f"{len(chunk.get('provenance') or [])} provenance regions"
                    )
                    if caption_text:
                        st.write(caption_text)

        ids = [str(item["id"]) for item in hierarchy_details]
        current = st.session_state.selected_hierarchical_chunk_id
        selected_hierarchy_id = st.selectbox(
            "Selected natural chunk",
            ids,
            index=ids.index(current) if current in ids else 0,
            format_func=lambda value: next(
                f"#{item['ordinal']} · {item['kind']} · {' → '.join(str(v) for v in item.get('section_path') or [])}"
                for item in hierarchy_details
                if str(item["id"]) == value
            ),
            key="structure_hierarchy_chunk",
        )
        st.session_state.selected_hierarchical_chunk_id = selected_hierarchy_id
        detail = next(item for item in hierarchy_details if str(item["id"]) == selected_hierarchy_id)
        render_chunk_detail(detail, show_embedding_text=False)
        st.caption("Hierarchical inspection chunks are not indexed and should have no vector ID.")
    else:
        st.warning("No hierarchical chunks are available for this page.")

with hybrid_tab:
    st.markdown("## Embedding representations")
    st.caption(
        "HybridChunker merges compatible structure while respecting an 800-token contextualized limit. "
        "A smaller chunk can be the correct semantic unit."
    )
    filter_columns = st.columns(4)
    with filter_columns[0]:
        kind = st.selectbox(
            "Kind",
            ["All", "text", "list", "table", "picture", "formula", "code", "mixed"],
            key="hybrid_kind_filter",
        )
        page_filter = st.selectbox(
            "Page",
            ["All", *list(range(1, 11))],
            key="hybrid_page_filter",
        )
    with filter_columns[1]:
        classification = st.selectbox(
            "Source / derived / mixed",
            ["All", "source", "derived", "mixed"],
            key="hybrid_classification_filter",
        )
        precise_filter = st.selectbox(
            "Precise provenance",
            ["Any", "Yes", "No"],
            key="hybrid_precise_filter",
        )
    with filter_columns[2]:
        table_filter = st.selectbox(
            "Has table",
            ["Any", "Yes", "No"],
            key="hybrid_table_filter",
        )
        picture_filter = st.selectbox(
            "Has picture",
            ["Any", "Yes", "No"],
            key="hybrid_picture_filter",
        )
    with filter_columns[3]:
        indexed_filter = st.selectbox(
            "Indexed",
            ["Any", "Yes", "No"],
            key="hybrid_indexed_filter",
        )

    try:
        response = client.list_chunks(
            docling_run_id,
            chunk_role="vector_index",
            kind=None if kind == "All" else kind,
            page=None if page_filter == "All" else int(page_filter),
            limit=100,
            include_text=False,
        )
        candidates = list(response.get("items") or [])
        details_for_advanced: dict[str, dict[str, Any]] = {}
        if table_filter != "Any" or picture_filter != "Any":
            details_for_advanced = {
                str(item["id"]): cached_chunk(client, str(item["id"])) for item in candidates
            }
    except ApiClientError as error:
        render_api_error(error, context="Hybrid chunks unavailable")
        candidates = []
        details_for_advanced = {}

    def keep(item: dict[str, Any]) -> bool:
        if classification != "All" and item.get("content_classification") != classification:
            return False
        if precise_filter != "Any" and bool(item.get("precise_provenance_available")) != (
            precise_filter == "Yes"
        ):
            return False
        if indexed_filter != "Any" and bool(item.get("vector_id")) != (indexed_filter == "Yes"):
            return False
        detail = details_for_advanced.get(str(item["id"]), {})
        if table_filter != "Any" and bool(detail.get("table_ref")) != (table_filter == "Yes"):
            return False
        if picture_filter != "Any" and bool(detail.get("picture_ref")) != (
            picture_filter == "Yes"
        ):
            return False
        return True

    filtered = [item for item in candidates if keep(item)]

    st.markdown("### Demonstration shortcuts")
    shortcut_columns = st.columns(5)
    shortcut_specs = [
        ("REQ-207 table", "REQ-207", 5),
        ("REQ-209 table", "REQ-209", 5),
        ("Page 8 OCR", "recovery window", 8),
        ("Page 9 formula", "M _ {", 9),
        ("Page 9 code", "enforce_thermal_margin", 9),
    ]

    def find_chunk(needle: str, target_page: int) -> str | None:
        data = client.list_chunks(
            docling_run_id,
            chunk_role="vector_index",
            page=target_page,
            limit=100,
            include_text=True,
        )
        match = next(
            (
                item
                for item in data.get("items") or []
                if needle.lower()
                in f"{item.get('raw_text') or ''}\n{item.get('embedding_text') or ''}".lower()
            ),
            None,
        )
        return str(match["id"]) if match else None

    for column, (label, needle, target_page) in zip(
        shortcut_columns, shortcut_specs, strict=True
    ):
        with column:
            if st.button(label, key=f"shortcut_{target_page}_{needle}", width="stretch"):
                try:
                    found = find_chunk(needle, target_page)
                    if found:
                        st.session_state.selected_hybrid_chunk_id = found
                        st.rerun()
                    st.warning("No matching stored chunk was returned by the API.")
                except ApiClientError as error:
                    render_api_error(error)

    page_size = 8
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    if int(st.session_state.get("hybrid_result_page", 1)) > total_pages:
        st.session_state.hybrid_result_page = 1
    list_page = st.number_input(
        "Result page",
        min_value=1,
        max_value=total_pages,
        value=1,
        key="hybrid_result_page",
    )
    page_items = filtered[(int(list_page) - 1) * page_size : int(list_page) * page_size]
    st.caption(f"{len(filtered)} matching vector chunks · showing at most {page_size} summaries")
    for item in page_items:
        st.markdown(
            badge(f"#{item.get('ordinal')}", "violet")
            + badge(str(item.get("kind")), "cyan")
            + badge(page_range(item), "cyan")
            + badge(str(item.get("content_classification")), "amber" if item.get("content_classification") != "source" else "cyan"),
            unsafe_allow_html=True,
        )

    selected_id = st.session_state.selected_hybrid_chunk_id
    selectable_ids = [str(item["id"]) for item in filtered]
    if selectable_ids:
        chosen = st.selectbox(
            "Selected Hybrid chunk",
            selectable_ids,
            index=selectable_ids.index(selected_id) if selected_id in selectable_ids else 0,
            format_func=lambda value: next(
                f"#{item['ordinal']} · {item['kind']} · {page_range(item)}"
                for item in filtered
                if str(item["id"]) == value
            ),
            key="structure_hybrid_chunk",
        )
        st.session_state.selected_hybrid_chunk_id = chosen
        selected_id = chosen

    if selected_id:
        try:
            selected = cached_chunk(client, str(selected_id))
            count = int(selected.get("contextualized_token_count") or 0)
            maximum = int(selected.get("max_token_count") or 800)
            st.progress(
                min(1.0, count / maximum if maximum else 0),
                text=f"Token budget · {count} / {maximum}",
            )
            render_chunk_detail(selected, show_embedding_text=True)
            if selected.get("table_ref"):
                st.info(
                    "Table relationships below are the API's preserved serialization. The UI does not reconstruct them."
                )
            artifact_ref = selected.get("picture_ref") or selected.get("table_ref")
            if artifact_ref:
                artifacts = cached_artifacts(client, docling_run_id)
                artifact = next(
                    (item for item in artifacts if item.get("doc_item_ref") == artifact_ref), None
                )
                if artifact:
                    image = cached_artifact_bytes(client, str(artifact["id"]))
                    st.image(image, caption=f"Registered source artifact · {artifact_ref}", width="stretch")
        except ApiClientError as error:
            render_api_error(error, context="Chunk detail unavailable")
    elif not filtered:
        st.warning("No Hybrid chunks match the current filters.")
