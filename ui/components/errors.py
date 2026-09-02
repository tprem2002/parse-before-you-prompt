"""Consistent safe error presentation for the UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import ApiClientError


_ACTIONS = {
    "api_unavailable": "Start FastAPI, then use Refresh status in the sidebar.",
    "dependency_unavailable": "Check PostgreSQL, Chroma, and model configuration before retrying.",
    "model_configuration_unavailable": "Complete the server-side model configuration, then refresh status.",
    "run_not_indexed": "Select an indexed run or process the document with Auto/Require indexing.",
    "processing_run_not_found": "Re-select the document and its persisted processing runs.",
    "artifact_not_found": "Refresh the artifact list and select another registered artifact.",
    "artifact_content_unavailable": "The registered artifact is missing; inspect the processing run warning state.",
    "precise_provenance_unavailable": "Use the broad page range. No exact overlay exists for this evidence.",
    "granite_docling_deferred": "Use Baseline or Docling standard. Granite-Docling remains deferred.",
    "evaluation_conflict": "Wait for the active evaluation to finish, then refresh its persisted status.",
    "evaluation_configuration_unavailable": "Verify both indexed runs and the server-side Azure configuration.",
    "invalid_ground_truth": "Inspect demo/ground_truth.json; malformed entries are never repaired silently.",
    "upstream_timeout": "Wait briefly, then submit the same question again if no result appeared.",
    "invalid_upstream_response": "Retry once. If it repeats, keep the request ID for troubleshooting.",
}


def render_api_error(error: ApiClientError, *, context: str | None = None) -> None:
    """Render only the shared safe envelope—never the raw response or traceback."""

    title = context or error.title
    st.error(f"**{title}**\n\n{error.message}")
    left, right = st.columns(2)
    left.caption(f"Code: `{error.code}`")
    right.caption(f"Request ID: `{error.request_id}`")
    action = _ACTIONS.get(error.code)
    if action:
        st.info(action, icon="➡️")
    if error.details:
        with st.expander("Safe development details", expanded=False):
            _render_safe_details(error.details)


def _render_safe_details(details: dict[str, Any]) -> None:
    """Present API-sanitized diagnostics without creating an unbounded JSON dump."""

    for key, value in list(details.items())[:20]:
        if isinstance(value, (str, int, float, bool)) or value is None:
            st.text(f"{key}: {value}")
        elif isinstance(value, list):
            st.text(f"{key}: {', '.join(str(item) for item in value[:20])}")
        else:
            st.text(f"{key}: structured detail available")


def render_unexpected_error() -> None:
    """Keep unexpected UI failures readable without exposing internals."""

    st.error(
        "**This view could not be rendered.**\n\n"
        "Refresh the page. If the problem persists, verify the selected document and runs."
    )
