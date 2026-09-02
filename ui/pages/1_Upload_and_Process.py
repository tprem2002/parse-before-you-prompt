"""Upload a PDF and queue the two implemented processing representations."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from api_client import ApiClientError, get_api_client
from components.branding import (
    badge,
    page_header,
    render_sidebar,
    setup_page,
    status_tone,
)
from components.errors import render_api_error
from components.run_status import render_run_status
from state import (
    ACTIVE_RUN_STATUSES,
    cached_configuration,
    cached_health,
    clear_read_caches,
    select_document_and_runs,
    use_aurora_demo,
)


setup_page()
render_sidebar()
page_header(section="1 · Upload and process")
client = get_api_client()

st.markdown("## Configuration readiness")
try:
    token = int(st.session_state.status_refresh_token)
    health = cached_health(client, token)
    config = cached_configuration(client, token)
    services = health.get("services") or {}
    azure = config.get("azure_openai") or {}
    readiness = [
        ("API", health.get("status") == "healthy"),
        ("PostgreSQL", (services.get("postgresql") or {}).get("status") == "healthy"),
        ("Chroma", (services.get("chroma") or {}).get("status") == "healthy"),
        ("Embeddings", bool(azure.get("embedding_ready"))),
        ("Chat", bool(azure.get("chat_ready"))),
        ("RAG", bool(azure.get("rag_model_ready"))),
    ]
    columns = st.columns(6)
    for column, (label, ready) in zip(columns, readiness, strict=True):
        with column:
            st.markdown(badge(label, status_tone(ready)), unsafe_allow_html=True)
            st.caption("Ready" if ready else "Needs attention")
except ApiClientError as error:
    render_api_error(error, context="Readiness unavailable")

st.markdown("## Existing demo")
with st.container(border=True):
    left, right = st.columns([4, 1])
    with left:
        st.markdown("**Project Aurora Mission Readiness Assessment**")
        st.caption(
            "Ten controlled pages with multi-column text, tables, pictures, OCR, formula, "
            "code, and persisted provenance. Selection reuses existing work and does not process it."
        )
    with right:
        if st.button("Use existing indexed demo", key="upload_use_aurora", width="stretch"):
            try:
                if use_aurora_demo(client):
                    st.success("Existing completed runs selected.")
                    st.rerun()
                else:
                    st.warning("The complete Aurora demo is not currently available.")
            except ApiClientError as error:
                render_api_error(error)

st.markdown("## Upload")
uploaded = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
    key="upload_pdf_file",
    help="PDF only. The API validates content, size, and page count.",
    max_upload_size=25,
)
if uploaded is not None:
    file_columns = st.columns(2)
    file_columns[0].write(f"Filename: `{uploaded.name}`")
    file_columns[1].write(f"Size: {uploaded.size / 1024:,.1f} KB")
    if st.button("Upload PDF", key="upload_pdf_submit", type="primary"):
        try:
            response = client.upload_document(filename=uploaded.name, content=uploaded.getvalue())
            document = response["document"]
            st.session_state.last_uploaded_document = response
            select_document_and_runs(document)
            st.session_state.pop("sidebar_document_selector", None)
            st.session_state.pop("sidebar_baseline_run_selector", None)
            st.session_state.pop("sidebar_docling_run_selector", None)
            clear_read_caches()
            if response.get("reused"):
                st.info(str(response.get("message") or "Existing document reused."))
            else:
                st.success(str(response.get("message") or "PDF uploaded."))
        except ApiClientError as error:
            render_api_error(error, context="Upload failed")

upload_result = st.session_state.last_uploaded_document
if isinstance(upload_result, dict):
    document = upload_result.get("document") or {}
    result_columns = st.columns(4)
    result_columns[0].metric("Pages", document.get("page_count") or "—")
    result_columns[1].metric("Size", f"{int(document.get('file_size_bytes') or 0) / 1024:,.1f} KB")
    result_columns[2].metric("Duplicate", "Yes" if upload_result.get("duplicate") else "No")
    result_columns[3].metric("Reused", "Yes" if upload_result.get("reused") else "No")
    st.caption(f"SHA-256: `{document.get('sha256', 'unavailable')}`")

st.markdown("## Processing modes")
mode_columns = st.columns(3)
with mode_columns[0]:
    baseline_selected = st.checkbox(
        "Baseline text extraction",
        value=True,
        key="process_mode_baseline",
        help="PyMuPDF plain text and fixed token chunks.",
    )
with mode_columns[1]:
    docling_selected = st.checkbox(
        "Docling standard structure-aware parsing",
        value=True,
        key="process_mode_docling",
        help="Local high-quality conversion, structured chunks, and provenance.",
    )
with mode_columns[2]:
    st.checkbox(
        "Granite-Docling-258M — optional comparison, not implemented",
        value=False,
        disabled=True,
        key="process_mode_granite_disabled",
    )

index_label = st.segmented_control(
    "Indexing mode",
    ["Auto", "Skip indexing", "Require indexing"],
    default="Auto",
    key="process_index_mode",
    help=(
        "Auto indexes when Azure embeddings are ready. Skip performs local conversion only. "
        "Require rejects the request if indexing is unavailable."
    ),
)
index_mode = {
    "Auto": "auto",
    "Skip indexing": "skip",
    "Require indexing": "required",
}.get(str(index_label), "auto")

selected_document_id = st.session_state.selected_document_id
if st.button(
    "Start selected processing",
    key="start_selected_processing",
    type="primary",
    disabled=not selected_document_id or not (baseline_selected or docling_selected),
):
    requests = []
    if baseline_selected:
        requests.append("baseline")
    if docling_selected:
        requests.append("docling_standard")
    for pipeline in requests:
        try:
            run = client.start_processing(
                str(selected_document_id), pipeline_type=pipeline, index_mode=index_mode
            )
            run_id = str(run["id"])
            st.session_state.active_processing_runs[run_id] = run
            if pipeline == "baseline":
                st.session_state.selected_baseline_run_id = run_id
            else:
                st.session_state.selected_docling_run_id = run_id
            if run.get("reused"):
                st.info(str(run.get("message") or f"Reused {pipeline} run."))
            else:
                st.success(str(run.get("message") or f"Queued {pipeline} run."))
        except ApiClientError as error:
            render_api_error(error, context=f"Could not start {pipeline.replace('_', ' ')}")

st.info(
    "The high-quality local Docling path can be compute-intensive. You may navigate away; "
    "the run state is persisted and can be reopened later."
)
st.caption("No completion-time estimate is shown. Processing history remains available through the API.")


@st.fragment(run_every=2)
def processing_monitor() -> None:
    active: dict[str, dict] = st.session_state.active_processing_runs
    should_poll = [
        run_id for run_id, run in active.items() if run.get("status") in ACTIVE_RUN_STATUSES
    ]
    for run_id in should_poll:
        try:
            active[run_id] = client.get_processing_run(run_id)
        except ApiClientError as error:
            render_api_error(error, context="Processing status unavailable")
    if should_poll:
        st.session_state.last_processing_refresh = datetime.now(timezone.utc).isoformat()
    if active:
        st.markdown("## Processing activity")
        for run in active.values():
            render_run_status(run)
        if st.session_state.last_processing_refresh:
            st.caption(f"Last refreshed: {st.session_state.last_processing_refresh}")
        if st.button("Refresh runs now", key="manual_processing_refresh"):
            for run_id, run in list(active.items()):
                if run.get("status") in ACTIVE_RUN_STATUSES:
                    try:
                        active[run_id] = client.get_processing_run(run_id)
                    except ApiClientError as error:
                        render_api_error(error)
            st.rerun(scope="fragment")


processing_monitor()
