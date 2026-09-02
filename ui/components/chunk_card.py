"""Structure-aware chunk cards used by inspection and evidence views."""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.branding import badge


def page_range(chunk: dict[str, Any]) -> str:
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if start is None:
        return "No page"
    return f"Page {start}" if end in {None, start} else f"Pages {start}–{end}"


def render_chunk_summary(chunk: dict[str, Any]) -> None:
    tones = {"source": "cyan", "derived": "amber", "mixed": "violet"}
    classification = str(chunk.get("content_classification") or "unknown")
    st.markdown(
        "".join(
            [
                badge(str(chunk.get("kind") or "unknown"), "violet"),
                badge(classification, tones.get(classification, "amber")),
                badge(page_range(chunk), "cyan"),
                badge(
                    "precise provenance" if chunk.get("precise_provenance_available") else "coarse provenance",
                    "cyan" if chunk.get("precise_provenance_available") else "amber",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )


def render_chunk_detail(chunk: dict[str, Any], *, show_embedding_text: bool = True) -> None:
    render_chunk_summary(chunk)
    headings = [str(item) for item in chunk.get("section_path") or []]
    if headings:
        st.caption("Heading path")
        st.write(" → ".join(headings))
    metadata = st.columns(4)
    metadata[0].metric("Raw tokens", int(chunk.get("raw_token_count") or 0))
    metadata[1].metric("Context tokens", int(chunk.get("contextualized_token_count") or 0))
    metadata[2].metric("Provenance", len(chunk.get("provenance") or []))
    metadata[3].metric("Indexed", "Yes" if chunk.get("vector_id") else "No")

    st.markdown("**Source-oriented raw text**")
    st.code(str(chunk.get("raw_text") or "No textual content was stored."), language=None)
    if show_embedding_text:
        st.markdown("**Contextualized text sent to the embedding model**")
        st.caption("Retrieval representation; not a verbatim quotation.")
        st.code(str(chunk.get("embedding_text") or "No contextualized text was stored."), language=None)

    captions = chunk.get("captions") or []
    if captions:
        st.markdown("**Source captions**")
        for caption in captions:
            text = caption.get("text") if isinstance(caption, dict) else caption
            st.info(f"Direct source evidence — {text}", icon="📄")

    components = (chunk.get("chunk_metadata") or {}).get("picture_components") or []
    for component in components:
        if not isinstance(component, dict):
            continue
        classifications = component.get("classification") or []
        if classifications and isinstance(classifications[0], dict):
            top = classifications[0]
            st.caption(
                f"Picture classification: {top.get('class_name', 'unknown')} · "
                f"{float(top.get('confidence') or 0):.3f}"
            )
        source_captions = component.get("source_captions") or []
        for caption in source_captions:
            text = caption.get("text") if isinstance(caption, dict) else caption
            st.info(f"Direct source evidence — {text}", icon="📄")
        description = component.get("derived_description") or component.get("generated_description")
        if isinstance(description, dict):
            description = description.get("text")
        if description:
            st.warning(
                "Derived visual description — generated locally from the source image\n\n"
                + str(description),
                icon="✨",
            )

    with st.expander("References and provenance", expanded=False):
        st.write("Docling item references")
        refs = chunk.get("doc_item_refs") or []
        st.code("\n".join(str(item) for item in refs) or "None", language=None)
        st.write(
            {
                "table_ref": chunk.get("table_ref"),
                "picture_ref": chunk.get("picture_ref"),
                "header_repetition": chunk.get("header_repetition_status") or "not applicable / not observed",
                "overflow": bool(chunk.get("overflow")),
            }
        )
