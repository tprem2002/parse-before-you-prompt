"""Shared page configuration, header, status sidebar, and visual primitives."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import streamlit as st

from api_client import ApiClientError, JsonObject, get_api_client
from components.errors import render_api_error
from state import (
    cached_configuration,
    cached_document,
    cached_documents,
    cached_health,
    clear_read_caches,
    initialize_state,
    select_document_and_runs,
)


_CSS_PATH = Path(__file__).resolve().parents[1] / "styles" / "app.css"


def setup_page() -> None:
    """Configure a page before its first rendered Streamlit element."""

    st.set_page_config(
        page_title="Parse Before You Prompt",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.html(_CSS_PATH)
    initialize_state()


def page_header(*, section: str | None = None) -> None:
    section_line = f"<div class='pbtp-kicker'>{section}</div>" if section else ""
    st.markdown(
        f"""
        <div class="pbtp-header">
          {section_line}
          <h1>Parse Before You Prompt</h1>
          <p class="pbtp-subtitle">Document Intelligence for Traceable RAG</p>
          <p class="pbtp-thesis">Parse accurately. Retrieve structurally. Answer with evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str, tone: str = "cyan") -> str:
    """Return controlled application-owned badge HTML."""

    safe_label = (
        label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    allowed = tone if tone in {"cyan", "violet", "amber", "green", "red"} else "cyan"
    return f'<span class="pbtp-badge pbtp-{allowed}">{safe_label}</span>'


def status_tone(value: bool | str | None) -> str:
    if value is True or value in {"healthy", "completed", "indexed", "ready"}:
        return "green"
    if value is False or value in {"unhealthy", "failed", "error"}:
        return "red"
    return "amber"


def metric_cards(items: Sequence[tuple[str, Any, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items, strict=True):
        with column:
            st.markdown(
                "<div class='pbtp-card'>"
                f"<div class='pbtp-card-label'>{label}</div>"
                f"<div class='pbtp-card-value'>{value}</div>"
                f"<div class='pbtp-card-note'>{note}</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def _short_identifier(value: str | None) -> str:
    return f"{value[:8]}…" if value else "Not selected"


def render_sidebar() -> tuple[JsonObject | None, JsonObject | None]:
    """Render safe global health, selection, and navigation guidance."""

    client = get_api_client()
    token = int(st.session_state.status_refresh_token)
    health: JsonObject | None = None
    configuration: JsonObject | None = None
    documents: list[JsonObject] = []
    with st.sidebar:
        st.markdown("### System status")
        try:
            health = cached_health(client, token)
            configuration = cached_configuration(client, token)
            documents = cached_documents(client, token)
            st.session_state.configuration_status = configuration
            st.session_state.api_health_cache_timestamp = datetime.now(timezone.utc).isoformat()
            services = health.get("services") or {}
            api_ok = health.get("status") == "healthy"
            postgres_ok = (services.get("postgresql") or {}).get("status") == "healthy"
            chroma_ok = (services.get("chroma") or {}).get("status") == "healthy"
            azure = configuration.get("azure_openai") or {}
            rows = [
                ("API", api_ok),
                ("PostgreSQL", postgres_ok),
                ("Chroma", chroma_ok),
                ("Embeddings", bool(azure.get("embedding_ready"))),
                ("Chat", bool(azure.get("chat_ready"))),
                ("RAG", bool(azure.get("rag_model_ready"))),
            ]
            for label, ready in rows:
                st.markdown(
                    f"{badge('● ' + label, status_tone(ready))} "
                    f"{'Ready' if ready else 'Needs attention'}",
                    unsafe_allow_html=True,
                )
        except ApiClientError as error:
            render_api_error(error, context="Status unavailable")

        if st.button("Refresh status", key="sidebar_refresh_status", width="stretch"):
            st.session_state.status_refresh_token += 1
            clear_read_caches()
            st.rerun()

        st.divider()
        st.markdown("### Demo selection")
        document_detail: JsonObject | None = None
        if documents:
            document_ids = [str(item["id"]) for item in documents]
            current = st.session_state.selected_document_id
            index = document_ids.index(current) if current in document_ids else 0
            selected_id = st.selectbox(
                "Document",
                document_ids,
                index=index,
                format_func=lambda value: next(
                    str(item.get("display_name") or item.get("filename"))
                    for item in documents
                    if str(item["id"]) == value
                ),
                key="sidebar_document_selector",
            )
            if selected_id != current:
                try:
                    select_document_and_runs(cached_document(client, selected_id))
                    st.session_state.pop("sidebar_baseline_run_selector", None)
                    st.session_state.pop("sidebar_docling_run_selector", None)
                    st.rerun()
                except ApiClientError as error:
                    render_api_error(error)
            try:
                document_detail = cached_document(client, selected_id)
            except ApiClientError as error:
                render_api_error(error)

        if document_detail:
            runs = list(document_detail.get("processing_runs") or [])

            def run_selector(label: str, pipeline: str, state_key: str, widget_key: str) -> None:
                options = [
                    str(run["id"])
                    for run in runs
                    if run.get("pipeline_type") == pipeline and run.get("status") != "failed"
                ]
                if not options:
                    st.caption(f"{label}: no usable run")
                    return
                current_run = st.session_state.get(state_key)
                selected_run = st.selectbox(
                    label,
                    options,
                    index=options.index(current_run) if current_run in options else 0,
                    format_func=lambda value: next(
                        f"{run.get('status')} · {value[:8]}…"
                        for run in runs
                        if str(run["id"]) == value
                    ),
                    key=widget_key,
                )
                st.session_state[state_key] = selected_run

            run_selector(
                "Baseline run",
                "baseline",
                "selected_baseline_run_id",
                "sidebar_baseline_run_selector",
            )
            run_selector(
                "Docling run",
                "docling_standard",
                "selected_docling_run_id",
                "sidebar_docling_run_selector",
            )

        st.caption(f"Document: `{_short_identifier(st.session_state.selected_document_id)}`")
        st.caption(f"Baseline: `{_short_identifier(st.session_state.selected_baseline_run_id)}`")
        st.caption(f"Docling: `{_short_identifier(st.session_state.selected_docling_run_id)}`")

        if configuration:
            azure = configuration.get("azure_openai") or {}
            st.markdown("#### Model readiness")
            st.caption(
                f"Embedding deployment: {'configured' if azure.get('embedding_deployment_configured') else 'missing'}"
            )
            st.caption(f"Vector dimension: {_selected_vector_dimension(client)}")
            st.caption(
                f"Chat deployment: {'configured' if azure.get('chat_deployment_configured') else 'missing'}"
            )
            st.caption(f"RAG: {'ready' if azure.get('rag_model_ready') else 'not ready'}")

        st.divider()
        st.markdown("### Presentation path")
        st.page_link("Home.py", label="Overview", icon="🏠", width="stretch")
        st.page_link("pages/2_Parsing_Comparison.py", label="Compare page 8", icon="🔎", width="stretch")
        st.page_link("pages/3_Structure_and_Chunks.py", label="Inspect chunks", icon="🧩", width="stretch")
        st.page_link("pages/4_Ask_and_Verify.py", label="Ask and verify", icon="✅", width="stretch")

    return health, configuration


def _selected_vector_dimension(client) -> str:  # type: ignore[no-untyped-def]
    for key in ("selected_docling_run_id", "selected_baseline_run_id"):
        run_id = st.session_state.get(key)
        if not run_id:
            continue
        try:
            run = cached_run_for_sidebar(client, str(run_id))
        except ApiClientError:
            continue
        dimension = (run.get("configuration_summary") or {}).get("vector_dimension")
        if dimension:
            return str(dimension)
    return "not available"


@st.cache_data(ttl=20, show_spinner=False)
def cached_run_for_sidebar(_client, run_id: str) -> JsonObject:  # type: ignore[no-untyped-def]
    return _client.get_processing_run(run_id)


def require_comparison_selection() -> tuple[str, str, str] | None:
    values = (
        st.session_state.selected_document_id,
        st.session_state.selected_baseline_run_id,
        st.session_state.selected_docling_run_id,
    )
    if all(values):
        return str(values[0]), str(values[1]), str(values[2])
    st.warning(
        "Select a document with completed Baseline and Docling standard runs before opening this comparison."
    )
    st.page_link(
        "pages/1_Upload_and_Process.py",
        label="Choose an existing demo or upload a PDF",
        icon="→",
        width="stretch",
    )
    return None
