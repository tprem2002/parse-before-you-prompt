"""Executive landing page for the Parse Before You Prompt demonstration."""

from __future__ import annotations

import streamlit as st

from api_client import ApiClientError, get_api_client
from components.branding import metric_cards, render_sidebar, setup_page
from components.errors import render_api_error
from state import cached_documents, cached_run, use_aurora_demo


setup_page()
health, configuration = render_sidebar()
client = get_api_client()

st.markdown(
    """
    <section class="pbtp-hero">
      <div class="eyebrow">Document Intelligence for Traceable RAG</div>
      <h1>Parse Before You Prompt</h1>
      <p>Why document intelligence is the hidden layer of reliable RAG</p>
      <div class="pbtp-thesis">Parse accurately. Retrieve structurally. Answer with evidence.</div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pbtp-flow">
      <div class="pbtp-flow-step">Document</div>
      <div class="pbtp-flow-step">Parsing</div>
      <div class="pbtp-flow-step">Structure-aware chunks</div>
      <div class="pbtp-flow-step">Retrieval</div>
      <div class="pbtp-flow-step">Evidence-grounded answer</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    documents = cached_documents(client, int(st.session_state.status_refresh_token))
    baseline_indexed = int((configuration or {}).get("indexed_baseline_run_count") or 0)
    docling_indexed = int(
        (configuration or {}).get("indexed_docling_standard_run_count") or 0
    )
    collections = int((configuration or {}).get("chroma_collection_count") or 0)
    provenance = 0
    run_id = st.session_state.selected_docling_run_id
    if run_id:
        provenance = int(cached_run(client, str(run_id)).get("provenance_count") or 0)
    metric_cards(
        [
            ("Documents", len(documents), "Live API inventory"),
            ("Baseline indexed", baseline_indexed, "PyMuPDF representation"),
            ("Docling indexed", docling_indexed, "Structured representation"),
            ("Provenance regions", provenance, "Selected Docling run"),
            ("Vector collections", collections, "Pipeline-isolated"),
        ]
    )
except ApiClientError as error:
    render_api_error(error, context="System summary unavailable")

st.markdown("## Start the demonstration")
actions = st.columns(4)
with actions[0]:
    if st.button("Use Project Aurora demo", key="home_use_aurora", width="stretch"):
        try:
            if use_aurora_demo(client):
                st.success("Project Aurora and both completed indexed runs are selected.")
                st.rerun()
            else:
                st.warning("Project Aurora or one of its completed runs is not available.")
        except ApiClientError as error:
            render_api_error(error)
with actions[1]:
    st.page_link(
        "pages/1_Upload_and_Process.py",
        label="Upload another PDF",
        icon="📄",
        width="stretch",
    )
with actions[2]:
    st.page_link(
        "pages/2_Parsing_Comparison.py",
        label="Open parsing comparison",
        icon="🔎",
        width="stretch",
    )
with actions[3]:
    st.page_link(
        "pages/4_Ask_and_Verify.py",
        label="Ask and verify",
        icon="✅",
        width="stretch",
    )

left, right = st.columns([1.2, 1])
with left:
    st.markdown("## The demonstration thesis")
    st.info(
        "A model can only retrieve what the ingestion pipeline preserved. This demo holds "
        "the Azure embedding deployment, Chroma settings, top-k, chat deployment, and "
        "answer prompt constant—then changes the document representation."
    )
with right:
    st.markdown("## Measured, not marketing")
    st.warning(
        "This project does not claim Docling is universally the best parser. Results are "
        "measured against the controlled Project Aurora document. Sanitized reference "
        "metrics are included with the public companion repository."
    )
