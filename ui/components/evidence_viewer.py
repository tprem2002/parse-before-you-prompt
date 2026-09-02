"""Claim-to-source evidence viewer for an interactive selected citation."""

from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import ApiClientError, get_api_client
from components.branding import badge
from components.chunk_card import page_range, render_chunk_detail
from components.errors import render_api_error
from state import cached_chunk, cached_evidence_image


def render_selected_citation(selection: dict[str, Any] | None) -> None:
    if not selection:
        st.info("Select a citation chip above to trace a claim to its retrieved evidence.")
        return
    citation = selection.get("citation") or {}
    claim = str(selection.get("claim") or "Selected claim")
    pipeline = str(selection.get("pipeline") or "")
    chunk_id = str(citation.get("chunk_id") or "")
    if not chunk_id:
        st.warning("The selected citation did not resolve to a retrieved chunk.")
        return

    client = get_api_client()
    try:
        chunk = cached_chunk(client, chunk_id)
    except ApiClientError as error:
        render_api_error(error, context="Citation evidence unavailable")
        return

    st.markdown("### Citation evidence")
    st.markdown(
        badge(str(citation.get("evidence_id") or "Evidence"), "violet")
        + badge(page_range(chunk), "cyan")
        + badge(
            str(chunk.get("content_classification") or "unknown"),
            "amber" if chunk.get("is_derived_content") else "cyan",
        ),
        unsafe_allow_html=True,
    )
    render_chunk_detail(chunk, show_embedding_text=False)

    pages = [int(item) for item in chunk.get("available_evidence_pages") or []]
    refs = [str(item) for item in chunk.get("doc_item_refs") or []]
    if pipeline == "baseline" or not chunk.get("precise_provenance_available"):
        st.warning(
            "Baseline evidence stops at a broad page range. Precise bounding-box provenance "
            "is unavailable, so no overlay is requested or invented."
        )
        chain = f"{claim} → {citation.get('evidence_id')} → chunk {chunk_id[:8]}… → {page_range(chunk)}"
        st.code(chain, language=None)
        return

    page = pages[0] if pages else chunk.get("page_start")
    chain = (
        f"{claim} → {citation.get('evidence_id')} → chunk {chunk_id[:8]}… → "
        f"{refs[0] if refs else 'Docling item'} → page {page} → highlighted source region"
    )
    st.code(chain, language=None)
    st.caption(
        f"Application-resolved provenance: {len(chunk.get('provenance') or [])} regions. "
        "The chat model did not generate page numbers or coordinates."
    )
    if page is not None:
        try:
            image = cached_evidence_image(client, chunk_id, int(page))
            st.image(image, caption=f"Highlighted source evidence · page {page}", width="stretch")
        except ApiClientError as error:
            render_api_error(error, context="Precise overlay unavailable")
