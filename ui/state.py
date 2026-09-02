"""Shared Streamlit state, cached reads, and Project Aurora selection."""

from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import ApiClient, JsonObject, get_api_client


AURORA_SHA256 = "1425eaee4597c0013b1c4933189e57ecfa14557ad0ac086ef06c93c39e412c34"
TERMINAL_RUN_STATUSES = {"completed", "partial", "failed"}
ACTIVE_RUN_STATUSES = {"queued", "running"}

_DEFAULTS: dict[str, Any] = {
    "selected_document_id": None,
    "selected_baseline_run_id": None,
    "selected_docling_run_id": None,
    "selected_page": 8,
    "selected_hierarchical_chunk_id": None,
    "selected_hybrid_chunk_id": None,
    "selected_citation": None,
    "last_question": "What is the maximum recovery window?",
    "last_baseline_rag_response": None,
    "last_docling_rag_response": None,
    "active_evaluation_id": None,
    "last_evaluation_response": None,
    "evaluation_refresh_token": 0,
    "active_processing_runs": {},
    "api_health_cache_timestamp": None,
    "configuration_status": None,
    "ui_display_preferences": {
        "show_source_metadata": True,
        "show_precise_provenance": True,
        "page_image_mode": "Original page",
    },
    "status_refresh_token": 0,
    "last_uploaded_document": None,
    "last_processing_refresh": None,
}


def initialize_state() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, dict) else value


@st.cache_data(ttl=5, show_spinner=False)
def cached_health(_client: ApiClient, refresh_token: int = 0) -> JsonObject:
    del refresh_token
    return _client.health()


@st.cache_data(ttl=5, show_spinner=False)
def cached_configuration(_client: ApiClient, refresh_token: int = 0) -> JsonObject:
    del refresh_token
    return _client.configuration_status()


@st.cache_data(ttl=10, show_spinner=False)
def cached_documents(_client: ApiClient, refresh_token: int = 0) -> list[JsonObject]:
    del refresh_token
    return _client.list_documents(limit=100)


@st.cache_data(ttl=20, show_spinner=False)
def cached_document(_client: ApiClient, document_id: str) -> JsonObject:
    return _client.get_document(document_id)


@st.cache_data(ttl=20, show_spinner=False)
def cached_run(_client: ApiClient, run_id: str) -> JsonObject:
    return _client.get_processing_run(run_id)


@st.cache_data(ttl=30, show_spinner=False)
def cached_artifacts(_client: ApiClient, run_id: str) -> list[JsonObject]:
    return _client.list_artifacts(run_id)


@st.cache_data(ttl=120, show_spinner=False)
def cached_artifact_bytes(_client: ApiClient, artifact_id: str) -> bytes:
    return _client.get_artifact_content(artifact_id).content


@st.cache_data(ttl=120, show_spinner=False)
def cached_chunk(_client: ApiClient, chunk_id: str) -> JsonObject:
    return _client.get_chunk(chunk_id)


@st.cache_data(ttl=120, show_spinner=False)
def cached_evidence_image(
    _client: ApiClient, chunk_id: str, page_no: int | None
) -> bytes:
    return _client.get_evidence_image(chunk_id, page_no=page_no).content


def select_document_and_runs(document: JsonObject) -> tuple[str | None, str | None]:
    """Select the newest completed run for each implemented pipeline."""

    runs = list(document.get("processing_runs") or [])

    def choose(pipeline: str) -> str | None:
        completed = [
            run
            for run in runs
            if run.get("pipeline_type") == pipeline and run.get("status") == "completed"
        ]
        if not completed:
            return None
        completed.sort(key=lambda item: str(item.get("completed_at") or ""), reverse=True)
        return str(completed[0]["id"])

    baseline_id = choose("baseline")
    docling_id = choose("docling_standard")
    st.session_state.selected_document_id = str(document["id"])
    st.session_state.selected_baseline_run_id = baseline_id
    st.session_state.selected_docling_run_id = docling_id
    return baseline_id, docling_id


def use_aurora_demo(client: ApiClient | None = None) -> bool:
    """Resolve Aurora from the live document API and select existing runs only."""

    api = client or get_api_client()
    documents = api.list_documents(limit=100)
    document = next(
        (
            item
            for item in documents
            if item.get("sha256") == AURORA_SHA256
            or str(item.get("filename", "")).lower()
            == "project_aurora_mission_readiness_report.pdf"
        ),
        None,
    )
    if document is None:
        return False
    detail = api.get_document(str(document["id"]))
    baseline_id, docling_id = select_document_and_runs(detail)
    st.session_state.pop("sidebar_document_selector", None)
    st.session_state.pop("sidebar_baseline_run_selector", None)
    st.session_state.pop("sidebar_docling_run_selector", None)
    return baseline_id is not None and docling_id is not None


def clear_read_caches() -> None:
    cached_health.clear()
    cached_configuration.clear()
    cached_documents.clear()
    cached_document.clear()
    cached_run.clear()
    cached_artifacts.clear()
