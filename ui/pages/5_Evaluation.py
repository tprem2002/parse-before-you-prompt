"""Run, poll, inspect, and export the controlled Project Aurora evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from api_client import ApiClientError, get_api_client
from components.branding import page_header, render_sidebar, require_comparison_selection, setup_page
from components.errors import render_api_error


setup_page()
render_sidebar()
page_header(section="5 · Controlled evaluation")
selection = require_comparison_selection()
if selection is None:
    st.stop()
document_id, baseline_run_id, docling_run_id = selection
client = get_api_client()

# Allow a safe, shareable deep link to an existing evaluation. The identifier is
# still resolved exclusively through the FastAPI polling endpoint.
linked_evaluation_id = st.query_params.get("evaluation_id")
if linked_evaluation_id and not st.session_state.active_evaluation_id:
    st.session_state.active_evaluation_id = str(linked_evaluation_id)


def metric_value(summary: dict[str, Any], *path: str) -> float | None:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


st.markdown("## Evaluation configuration")
try:
    configuration = client.evaluation_configuration(
        document_id=document_id,
        baseline_processing_run_id=baseline_run_id,
        docling_processing_run_id=docling_run_id,
        top_k=5,
    )
except ApiClientError as error:
    render_api_error(error, context="Evaluation preflight unavailable")
    st.stop()

config = dict(configuration.get("configuration") or {})
first = st.columns(4)
first[0].metric("Questions", int(configuration.get("question_count") or 0))
first[1].metric("Answerable", int(configuration.get("answerable_count") or 0))
first[2].metric("Unsupported", int(configuration.get("unsupported_count") or 0))
first[3].metric("Pipeline cases", int(configuration.get("total_cases") or 0))
second = st.columns(4)
second[0].metric("top_k", int(config.get("top_k") or 0))
second[1].metric("Vector dimension", int(config.get("vector_dimension") or 0))
second[2].metric("Embedding", str(config.get("embedding_deployment") or "—"))
second[3].metric("Chat", str(config.get("chat_deployment") or "—"))
st.caption(
    f"Document {document_id[:8]}… · Baseline {baseline_run_id[:8]}… · "
    f"Docling {docling_run_id[:8]}… · Prompt {config.get('prompt_version')} · "
    f"Metrics {configuration.get('metric_definition_version')}"
)
st.caption(
    "This preflight reads fixed ground truth and PostgreSQL index metadata only. It creates no "
    "EvaluationRun and makes no Chroma query or Azure call."
)

st.markdown("## Run control")
confirmed = st.checkbox(
    "This will execute every ground-truth question against both indexed pipelines and will make live Azure model calls.",
    key="evaluation_live_confirmation",
)
if st.button(
    "Run controlled evaluation",
    key="evaluation_run_controlled",
    type="primary",
    width="stretch",
    disabled=not confirmed,
):
    try:
        response = client.start_evaluation(
            document_id=document_id,
            baseline_processing_run_id=baseline_run_id,
            docling_processing_run_id=docling_run_id,
            top_k=5,
        )
        st.session_state.active_evaluation_id = str(response["evaluation_id"])
        st.session_state.last_evaluation_response = response
        st.rerun()
    except ApiClientError as error:
        render_api_error(error, context="Evaluation could not start")


def render_scorecards(payload: dict[str, Any]) -> None:
    summaries = dict(payload.get("pipeline_summaries") or {})
    st.markdown("## Measured summary")
    for pipeline, label in (("baseline", "Baseline"), ("docling_standard", "Docling standard")):
        summary = dict(summaries.get(pipeline) or {})
        st.markdown(f"### {label}")
        metrics = [
            ("Recall@1", percent(metric_value(summary, "retrieval", "recall_at_1"))),
            ("Recall@3", percent(metric_value(summary, "retrieval", "recall_at_3"))),
            ("Recall@5", percent(metric_value(summary, "retrieval", "recall_at_5"))),
            ("MRR", f"{metric_value(summary, 'retrieval', 'mrr') or 0:.3f}"),
            ("Answer match", percent(metric_value(summary, "answer", "normalized_answer_match_rate"))),
            ("Table accuracy", percent(metric_value(summary, "answer", "table_question_accuracy"))),
            ("Citation page", percent(metric_value(summary, "citation", "citation_page_accuracy"))),
            ("Precise provenance", percent(metric_value(summary, "citation", "precise_provenance_availability_rate"))),
            ("Abstention", percent(metric_value(summary, "abstention", "unsupported_abstention_accuracy"))),
            ("Answerability", percent(metric_value(summary, "abstention", "answerability_decision_accuracy"))),
            ("Mean total latency", f"{metric_value(summary, 'latency_ms', 'total', 'mean') or 0:,.0f} ms"),
        ]
        for start in range(0, len(metrics), 4):
            columns = st.columns(min(4, len(metrics) - start))
            for column, (name, value) in zip(columns, metrics[start:start + 4], strict=True):
                column.metric(name, value)


def render_charts(payload: dict[str, Any]) -> None:
    summaries = dict(payload.get("pipeline_summaries") or {})
    rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    for pipeline, label in (("baseline", "Baseline"), ("docling_standard", "Docling standard")):
        summary = dict(summaries.get(pipeline) or {})
        rows.append({
            "Pipeline": label,
            "Recall@1": metric_value(summary, "retrieval", "recall_at_1"),
            "Recall@3": metric_value(summary, "retrieval", "recall_at_3"),
            "Recall@5": metric_value(summary, "retrieval", "recall_at_5"),
            "MRR": metric_value(summary, "retrieval", "mrr"),
        })
        answer_rows.append({
            "Pipeline": label,
            "Answer match": metric_value(summary, "answer", "normalized_answer_match_rate"),
            "Citation page": metric_value(summary, "citation", "citation_page_accuracy"),
            "Precise provenance": metric_value(summary, "citation", "precise_provenance_availability_rate"),
            "Abstention": metric_value(summary, "abstention", "unsupported_abstention_accuracy"),
        })
        latency_rows.append({
            "Pipeline": label,
            "Retrieval": metric_value(summary, "latency_ms", "retrieval", "mean"),
            "Generation": metric_value(summary, "latency_ms", "generation", "mean"),
        })
    left, right = st.columns(2)
    with left:
        st.markdown("#### Retrieval metrics")
        st.bar_chart(pd.DataFrame(rows).set_index("Pipeline"), y_label="Rate / MRR")
    with right:
        st.markdown("#### Answer, citation, and abstention")
        st.bar_chart(pd.DataFrame(answer_rows).set_index("Pipeline"), y_label="Rate")
    st.markdown("#### Mean retrieval and generation latency")
    st.bar_chart(pd.DataFrame(latency_rows).set_index("Pipeline"), y_label="Milliseconds")

    categories = list(payload.get("category_summaries") or [])
    category_rows = []
    for item in categories:
        retrieval = dict(item.get("retrieval") or {})
        category_rows.append({
            "Pipeline / kind": f"{item.get('pipeline_label')} · {item.get('expected_kind')} (n={item.get('question_count')})",
            "Recall@1": metric_value({"r": retrieval}, "r", "recall_at_1"),
            "Recall@3": metric_value({"r": retrieval}, "r", "recall_at_3"),
            "Recall@5": metric_value({"r": retrieval}, "r", "recall_at_5"),
            "MRR": metric_value({"r": retrieval}, "r", "mrr"),
        })
    if category_rows:
        st.markdown("#### Category retrieval diagnostics")
        st.bar_chart(pd.DataFrame(category_rows).set_index("Pipeline / kind"), y_label="Rate / MRR")


def render_results(payload: dict[str, Any]) -> None:
    results = list(payload.get("results") or [])
    if not results:
        return
    st.markdown("## Per-question results")
    filters = st.columns(4)
    pipelines = ["All", *sorted({str(item["pipeline_label"]) for item in results})]
    kinds = ["All", *sorted({str(item["expected_kind"]) for item in results})]
    selected_pipeline = filters[0].selectbox("Pipeline", pipelines, key="evaluation_filter_pipeline")
    selected_kind = filters[1].selectbox("Question kind", kinds, key="evaluation_filter_kind")
    selected_answerability = filters[2].selectbox("Answerability", ["All", "Answerable", "Unsupported"], key="evaluation_filter_answerable")
    selected_result = filters[3].selectbox("Result", ["All", "pass", "fail", "error"], key="evaluation_filter_result")
    flags = st.columns(3)
    misses_only = flags[0].checkbox("Retrieval misses only", key="evaluation_filter_miss")
    wrong_pages_only = flags[1].checkbox("Citation-page failures only", key="evaluation_filter_page")
    precise_only = flags[2].checkbox("Precise provenance only", key="evaluation_filter_precise")
    filtered = []
    for item in results:
        if selected_pipeline != "All" and item["pipeline_label"] != selected_pipeline:
            continue
        if selected_kind != "All" and item["expected_kind"] != selected_kind:
            continue
        if selected_answerability == "Answerable" and not item["answerable"]:
            continue
        if selected_answerability == "Unsupported" and item["answerable"]:
            continue
        if selected_result != "All" and item["result_status"] != selected_result:
            continue
        if misses_only and not item.get("ingestion_or_representation_miss"):
            continue
        if wrong_pages_only and item.get("citation_page_correct") is not False:
            continue
        if precise_only and not item.get("precise_provenance_available"):
            continue
        filtered.append(item)
    table_rows = [{
        "Case": f"{item['question_id']} · {item['pipeline_label']}",
        "Question": item["question"],
        "Answer": item.get("answer") or "—",
        "Expected": " · ".join(item.get("accepted_answers") or []) if item["answerable"] else "Structural abstention",
        "Kind": item["expected_kind"],
        "First relevant": item.get("first_relevant_rank"),
        "R@1": item.get("recall_at_1"), "R@3": item.get("recall_at_3"), "R@5": item.get("recall_at_5"),
        "Answer match": item.get("normalized_answer_match"),
        "Abstention": item.get("structural_abstention_correct"),
        "Citations": ", ".join(item.get("citation_ids") or []),
        "Pages": ", ".join(str(value) for value in item.get("cited_pages") or []),
        "Total ms": item.get("total_duration_ms"), "Result": item["result_status"],
    } for item in filtered]
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch", height=440)
    if not filtered:
        return
    labels = [f"{item['question_id']} · {item['pipeline_label']} · {item['question']}" for item in filtered]
    selected_label = st.selectbox("Question detail", labels, key="evaluation_question_detail")
    detail = filtered[labels.index(selected_label)]
    st.markdown(f"### {detail['question_id']} · {detail['pipeline_label']}")
    st.write(detail.get("answer") or "No answer returned.")
    st.caption(
        f"Expected: {' · '.join(detail.get('accepted_answers') or []) if detail['answerable'] else 'structural abstention'} · "
        f"Match: {detail.get('normalized_answer_match')} · Citation page: {detail.get('citation_page_correct')} · "
        f"Precise provenance: {detail.get('precise_provenance_available')}"
    )
    if detail.get("claims"):
        st.markdown("**Claims and application-resolved citations**")
        for claim in detail["claims"]:
            st.write(f"{claim.get('text')} — {', '.join(claim.get('citation_ids') or [])}")
    for hit in detail.get("ranked_retrieval") or []:
        title = f"{hit['evidence_id']} · rank {hit['rank']} · distance {float(hit['distance']):.4f} · relevant {hit['relevant']}"
        with st.expander(title):
            st.write({
                "chunk_id": hit["chunk_id"], "pages": hit.get("resolved_pages"),
                "page_overlap": hit.get("page_overlap"), "kind": hit.get("kind"),
                "expected_term_coverage": hit.get("expected_term_coverage"),
            })
            citation = next((item for item in detail.get("citation_resolution") or []
                             if item.get("evidence_id") == hit.get("evidence_id")), None)
            image = (citation or {}).get("evidence_image") if citation else None
            if image and st.button("Load evidence overlay", key=f"eval_overlay_{detail['question_id']}_{detail['pipeline']}_{hit['rank']}"):
                pages = list(image.get("available_pages") or [])
                try:
                    st.image(client.get_evidence_image(hit["chunk_id"], page_no=pages[0] if pages else None).content)
                except ApiClientError as error:
                    render_api_error(error, context="Evidence overlay unavailable")


def render_downloads(payload: dict[str, Any]) -> None:
    evaluation_id = str(payload["evaluation_id"])
    try:
        csv_payload = client.export_evaluation_csv(evaluation_id)
        json_payload = client.export_evaluation_json(evaluation_id)
    except ApiClientError as error:
        render_api_error(error, context="Evaluation exports unavailable")
        return
    columns = st.columns(2)
    columns[0].download_button("Download evaluation CSV", data=csv_payload.content,
                               file_name=csv_payload.filename or f"evaluation-{evaluation_id}.csv",
                               mime="text/csv", width="stretch")
    columns[1].download_button("Download evaluation JSON", data=json_payload.content,
                               file_name=json_payload.filename or f"evaluation-{evaluation_id}.json",
                               mime="application/json", width="stretch")


@st.fragment(run_every=2)
def evaluation_status() -> None:
    evaluation_id = st.session_state.active_evaluation_id
    if not evaluation_id:
        return
    try:
        current = client.get_evaluation(str(evaluation_id), include_results=True)
        st.session_state.last_evaluation_response = current
    except ApiClientError as error:
        render_api_error(error, context="Evaluation status unavailable")
        return
    status_value = str(current.get("status") or "unknown")
    st.markdown("## Progress")
    st.progress(float(current.get("progress_percent") or 0) / 100.0)
    st.caption(
        f"Status: {status_value} · Current question: {current.get('current_question_id') or '—'} · "
        f"Cases: {current.get('completed_case_count')}/{current.get('total_case_count')} · "
        f"Elapsed: {int(current.get('elapsed_duration_ms') or 0):,} ms"
    )
    if current.get("safe_error"):
        st.error(str((current["safe_error"] or {}).get("message") or "Evaluation failed safely."))
    if status_value in {"completed", "completed_with_failures"}:
        render_scorecards(current)
        render_charts(current)
        render_results(current)
        render_downloads(current)
    elif status_value in {"failed", "interrupted"}:
        st.warning("The evaluation did not produce a publishable completed benchmark. Partial cases remain available for diagnosis.")
        render_results(current)


evaluation_status()

st.markdown("## Methodology notice")
st.info(
    f"This is one pass over one synthetic 10-page document and {int(configuration.get('question_count') or 0)} questions. "
    "It is a descriptive benchmark, not a universal parser ranking. Baseline has only three broad chunks, "
    "so Recall@5 is less discriminating for that path. Citation integrity checks IDs and structure, not "
    "semantic entailment. Provenance supports verification but does not eliminate hallucination."
)
