"""Run the same RAG question against both document representations."""

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
from components.chunk_card import page_range
from components.errors import render_api_error
from components.evidence_viewer import render_selected_citation
from state import cached_chunk


setup_page()
render_sidebar()
page_header(section="4 · Ask and verify")
selection = require_comparison_selection()
if selection is None:
    st.stop()
_document_id, baseline_run_id, docling_run_id = selection
client = get_api_client()

examples = [
    "What is the maximum recovery window?",
    "What verification method and readiness status are recorded for REQ-207?",
    "What is Q4 readiness?",
    "What does the Sensor Gateway send data to?",
    "What was the contract award date?",
]


def use_example() -> None:
    st.session_state.last_question = st.session_state.ask_example_question


st.markdown("## Ask the same question of both representations")
example_columns = st.columns([4, 1])
with example_columns[0]:
    st.selectbox(
        "Suggested question",
        examples,
        key="ask_example_question",
        label_visibility="collapsed",
    )
with example_columns[1]:
    st.button(
        "Use example",
        key="ask_use_example",
        on_click=use_example,
        width="stretch",
    )

question_columns = st.columns([5, 1])
with question_columns[0]:
    question = st.text_input(
        "Question",
        key="last_question",
        placeholder="Ask a document-grounded question…",
    )
with question_columns[1]:
    top_k = st.number_input(
        "top_k",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="ask_top_k",
    )

st.caption(
    "Both calls receive the exact same question and top_k. No hidden keywords or expected answers are added."
)
if st.button(
    "Run both pipelines",
    key="ask_run_both",
    type="primary",
    width="stretch",
    disabled=not question.strip(),
):
    request_question = question.strip()
    st.session_state.selected_citation = None
    with st.spinner("Running Baseline RAG with the shared prompt…"):
        try:
            st.session_state.last_baseline_rag_response = client.run_rag_query(
                processing_run_id=baseline_run_id,
                question=request_question,
                top_k=int(top_k),
            )
        except ApiClientError as error:
            st.session_state.last_baseline_rag_response = None
            render_api_error(error, context="Baseline RAG failed")
    with st.spinner("Running Docling RAG with the same request…"):
        try:
            st.session_state.last_docling_rag_response = client.run_rag_query(
                processing_run_id=docling_run_id,
                question=request_question,
                top_k=int(top_k),
            )
        except ApiClientError as error:
            st.session_state.last_docling_rag_response = None
            render_api_error(error, context="Docling RAG failed")


def select_citation(
    response: dict[str, Any], citation: dict[str, Any], claim: str
) -> None:
    st.session_state.selected_citation = {
        "pipeline": response.get("pipeline_type"),
        "citation": citation,
        "claim": claim,
    }


def render_answer(response: dict[str, Any], *, label: str, tone: str) -> None:
    insufficient = bool(response.get("insufficient_evidence"))
    st.markdown(f"### {label}")
    st.markdown(
        badge(
            "Insufficient evidence" if insufficient else "Supported answer",
            "amber" if insufficient else "green",
        )
        + badge(
            f"Citation integrity: {response.get('citation_validation_status')}",
            "green" if response.get("citation_validation_status") == "valid" else "red",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(str(response.get("answer") or "No answer returned."))

    claims = list(response.get("claims") or [])
    resolved = {
        str(item.get("evidence_id")): item
        for item in response.get("resolved_citations") or []
    }
    if claims:
        st.markdown("**Claims and citations**")
        for claim_index, claim in enumerate(claims):
            claim_text = str(claim.get("text") or "")
            st.write(claim_text)
            citation_ids = list(claim.get("citation_ids") or [])
            citation_columns = st.columns(max(1, min(5, len(citation_ids))))
            for index, citation_id in enumerate(citation_ids):
                citation = resolved.get(str(citation_id))
                if citation and citation_columns[index % len(citation_columns)].button(
                    str(citation_id),
                    key=f"citation_{response.get('query_run_id')}_{claim_index}_{citation_id}",
                    help="Open the application-resolved evidence chain",
                ):
                    select_citation(response, citation, claim_text)
                    st.rerun()
    elif insufficient:
        st.caption("A structural abstention carries no factual claims or fabricated citations.")

    metric_rows = st.columns(4)
    metric_rows[0].metric("Actual hits", int(response.get("actual_hit_count") or 0))
    metric_rows[1].metric("Evidence tokens", int(response.get("total_evidence_tokens") or 0))
    metric_rows[2].metric("Retrieval", f"{int(response.get('retrieval_duration_ms') or 0):,} ms")
    metric_rows[3].metric("Generation", f"{int(response.get('generation_duration_ms') or 0):,} ms")
    st.caption(
        f"Total {int(response.get('total_duration_ms') or 0):,} ms · "
        f"Deployment {response.get('deployment') or 'not reported'} · "
        f"Service model {response.get('service_model') or 'not reported'} · "
        f"Prompt {response.get('prompt_version')} · Schema {response.get('schema_version')}"
    )
    st.caption(
        "Latency is shown as observed for this query; a shorter result is not universally better without context."
    )

    st.markdown("#### Retrieved evidence")
    for hit in response.get("retrieval_hits") or []:
        title = (
            f"{hit.get('evidence_id')} · rank {hit.get('rank')} · "
            f"Distance {float(hit.get('distance') or 0):.4f} · {page_range(hit)}"
        )
        with st.expander(title, expanded=False):
            classification = str(hit.get("source_classification") or "unknown")
            st.markdown(
                badge(str(hit.get("kind") or "unknown"), tone)
                + badge(
                    "Direct source evidence" if classification == "source" else "Derived visual description",
                    "cyan" if classification == "source" else "amber",
                )
                + badge(
                    "precise provenance"
                    if hit.get("precise_provenance_available")
                    else "broad page range",
                    "cyan" if hit.get("precise_provenance_available") else "amber",
                ),
                unsafe_allow_html=True,
            )
            headings = [str(item) for item in hit.get("section_path") or []]
            if headings:
                st.caption(" → ".join(headings))
            try:
                chunk = cached_chunk(client, str(hit["chunk_id"]))
                st.code(str(chunk.get("raw_text") or "No source-oriented text stored."), language=None)
            except ApiClientError as error:
                render_api_error(error, context="Retrieved chunk unavailable")
            st.caption(
                f"Distance: {hit.get('distance')} · provenance regions: "
                f"{hit.get('provenance_region_count')} · contextualized tokens: "
                f"{hit.get('contextualized_token_count')}"
            )


baseline_response = st.session_state.last_baseline_rag_response
docling_response = st.session_state.last_docling_rag_response
if baseline_response or docling_response:
    st.markdown("## Side-by-side result")
    baseline_column, docling_column = st.columns(2)
    with baseline_column:
        if isinstance(baseline_response, dict):
            render_answer(baseline_response, label="Baseline RAG", tone="amber")
        else:
            st.warning("No Baseline response is available.")
    with docling_column:
        if isinstance(docling_response, dict):
            render_answer(docling_response, label="Docling RAG", tone="violet")
        else:
            st.warning("No Docling response is available.")

    st.divider()
    render_selected_citation(st.session_state.selected_citation)
else:
    st.info(
        "Run both pipelines to compare what survived ingestion, what retrieval found, and how precisely each claim resolves to source evidence."
    )
