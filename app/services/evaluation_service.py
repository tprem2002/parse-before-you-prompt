"""Controlled, deterministic evaluation over the existing grounded RAG service."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import statistics
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import Settings, get_settings
from app.core.enums import PipelineType, ProcessingStatus
from app.core.errors import ApplicationError, ConfigurationError
from app.core.logging import get_logger
from app.db.models import Chunk, Document, EvaluationRun, ProcessingRun, QueryRun
from app.db.session import SessionLocal
from app.prompts import load_rag_prompt
from app.schemas.rag import RagResponse
from app.services.rag_service import GenerationValidationError, answer_question
from app.services.retrieval_service import RunNotIndexedError, inspect_run_readiness


logger = get_logger(__name__)

EVALUATION_VERSION = "parse-before-you-prompt-evaluation-v1"
METRIC_DEFINITION_VERSION = "parse-before-you-prompt-metrics-v1"
GROUND_TRUTH_RELATIVE_PATH = "demo/ground_truth.json"
LIVE_CONFIRMATION = "RUN_PROJECT_AURORA_EVALUATION"
TERMINAL_REUSABLE_STATUSES = {"completed", "completed_with_failures"}
ACTIVE_STATUSES = {"queued", "running"}
PIPELINE_LABELS = {
    PipelineType.BASELINE.value: "Baseline",
    PipelineType.DOCLING_STANDARD.value: "Docling standard",
}
STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "was",
}

METRIC_DEFINITIONS: dict[str, str] = {
    "recall_at_1": "Answerable questions with a strictly relevant hit at rank 1 divided by all answerable questions.",
    "recall_at_3": "Answerable questions with a strictly relevant hit in ranks 1-3 divided by all answerable questions.",
    "recall_at_5": "Answerable questions with a strictly relevant hit in ranks 1-5 divided by all answerable questions.",
    "mrr": "Mean reciprocal rank of the first strictly relevant hit over answerable questions; a representation miss contributes zero.",
    "normalized_answer_match_rate": "Answerable questions whose non-abstaining answer contains a normalized accepted alternative or all meaningful alternative tokens.",
    "table_question_accuracy": "Normalized answer-match rate restricted to answerable table questions.",
    "citation_integrity_rate": "Supported answers whose existing deterministic citation-ID validator passed; this is not semantic entailment.",
    "citation_page_accuracy": "Supported answers for which every factual claim has a valid citation resolving to an expected page.",
    "precise_provenance_availability_rate": "Supported answers with cited evidence backed by a stored page and complete bounding box.",
    "unsupported_abstention_accuracy": "Unsupported questions returning a nonempty structural abstention with no claims.",
    "supported_non_abstention_rate": "Answerable questions for which the model did not abstain.",
    "answerability_decision_accuracy": "Questions with the correct answerable/non-abstaining or unsupported/abstaining decision.",
}

CSV_COLUMNS = [
    "evaluation_id", "question_id", "pipeline", "question", "answerable",
    "expected_kind", "expected_pages", "accepted_answers", "requested_top_k",
    "actual_hit_count", "first_relevant_rank", "reciprocal_rank", "recall_at_1",
    "recall_at_3", "recall_at_5", "ingestion_or_representation_miss", "answer",
    "insufficient_evidence", "normalized_answer_match", "matched_accepted_answer",
    "claim_count", "citation_ids", "citation_integrity_valid", "citation_page_correct",
    "precise_provenance_available", "cited_pages", "cited_chunk_ids",
    "retrieval_duration_ms", "generation_duration_ms", "total_duration_ms",
    "evidence_token_count", "model_input_tokens", "model_output_tokens",
    "model_total_tokens", "validation_attempt_count", "failure_category", "result_status",
]


class GroundTruthValidationError(ValueError):
    """The repository-owned evaluation definition is malformed or ambiguous."""


class EvaluationConflictError(ApplicationError):
    """An incompatible or already-active controlled evaluation prevents execution."""


class EvaluationNotFoundError(LookupError):
    """The requested evaluation row does not exist."""


@dataclass(frozen=True, slots=True)
class GroundTruthQuestion:
    id: str
    question: str
    answerable: bool
    accepted_answers: tuple[str, ...]
    expected_pages: tuple[int, ...]
    expected_kind: str
    expected_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroundTruthSet:
    questions: tuple[GroundTruthQuestion, ...]
    sha256: str
    relative_path: str

    @property
    def answerable_count(self) -> int:
        return sum(item.answerable for item in self.questions)

    @property
    def unsupported_count(self) -> int:
        return len(self.questions) - self.answerable_count


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    document_id: uuid.UUID
    baseline_processing_run_id: uuid.UUID
    docling_processing_run_id: uuid.UUID
    top_k: int
    ground_truth: GroundTruthSet
    source_sha256: str
    fingerprint: str
    configuration: dict[str, Any]
    processing_metrics: dict[str, Any]

    @property
    def total_cases(self) -> int:
        return len(self.ground_truth.questions) * 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": str(self.document_id),
            "baseline_processing_run_id": str(self.baseline_processing_run_id),
            "docling_processing_run_id": str(self.docling_processing_run_id),
            "question_count": len(self.ground_truth.questions),
            "answerable_count": self.ground_truth.answerable_count,
            "unsupported_count": self.ground_truth.unsupported_count,
            "total_cases": self.total_cases,
            "top_k": self.top_k,
            "ground_truth_path": self.ground_truth.relative_path,
            "ground_truth_sha256": self.ground_truth.sha256,
            "source_sha256": self.source_sha256,
            "evaluation_version": EVALUATION_VERSION,
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "evaluation_fingerprint": self.fingerprint,
            "configuration": self.configuration,
            "processing_metrics": self.processing_metrics,
            "metric_definitions": METRIC_DEFINITIONS,
            "model_call_will_occur": False,
            "chroma_query_will_occur": False,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _hash_bytes(payload.encode("utf-8"))


def normalize_evaluation_text(value: str) -> str:
    """Normalize text once for deterministic relevance and answer matching."""

    text = unicodedata.normalize("NFKC", value).lower()
    text = text.translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
        "\u2014": "-", "\u2212": "-",
    }))
    text = re.sub(r"(?<!\w)(\d+(?:\.\d+)?)\s*%", r"\1 percent", text)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _contains_normalized_phrase(searchable: str, phrase: str) -> bool:
    haystack = searchable.split()
    needle = phrase.split()
    return bool(needle) and any(
        haystack[index:index + len(needle)] == needle
        for index in range(0, len(haystack) - len(needle) + 1)
    )


def _nonempty_string_list(value: Any, *, field: str, question_id: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise GroundTruthValidationError(f"{question_id}: {field} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise GroundTruthValidationError(f"{question_id}: {field}[{index}] must be a nonempty string")
        result.append(item.strip())
    return tuple(result)


def load_ground_truth(path: Path | None = None, *, maximum_page: int = 10) -> GroundTruthSet:
    """Load and strictly validate the fixed repository ground-truth file."""

    source_path = path or (_project_root() / GROUND_TRUTH_RELATIVE_PATH)
    raw = source_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundTruthValidationError("Ground truth must be valid UTF-8 JSON") from exc
    if not isinstance(payload, list):
        raise GroundTruthValidationError("Ground-truth root must be a JSON array")
    questions: list[GroundTruthQuestion] = []
    seen_ids: set[str] = set()
    abstention_terms = {"not found", "not stated", "insufficient evidence", "not provided", "cannot be determined"}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise GroundTruthValidationError(f"Entry {index} must be an object")
        question_id = item.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise GroundTruthValidationError(f"Entry {index}: id must be nonempty")
        question_id = question_id.strip()
        if question_id in seen_ids:
            raise GroundTruthValidationError(f"Duplicate question id: {question_id}")
        seen_ids.add(question_id)
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise GroundTruthValidationError(f"{question_id}: question must be nonempty")
        answerable = item.get("answerable")
        if type(answerable) is not bool:
            raise GroundTruthValidationError(f"{question_id}: answerable must be boolean")
        accepted = _nonempty_string_list(item.get("accepted_answers"), field="accepted_answers", question_id=question_id)
        pages_value = item.get("expected_pages")
        if not isinstance(pages_value, list):
            raise GroundTruthValidationError(f"{question_id}: expected_pages must be an array")
        pages: list[int] = []
        for page in pages_value:
            if type(page) is not int or page < 1 or page > maximum_page:
                raise GroundTruthValidationError(f"{question_id}: expected page must be between 1 and {maximum_page}")
            pages.append(page)
        expected_kind = item.get("expected_kind")
        if not isinstance(expected_kind, str) or not expected_kind.strip():
            raise GroundTruthValidationError(f"{question_id}: expected_kind must be nonempty")
        terms = _nonempty_string_list(item.get("expected_terms"), field="expected_terms", question_id=question_id)
        if answerable and (not accepted or not pages or not terms):
            raise GroundTruthValidationError(f"{question_id}: answerable entries require accepted answers, pages, and terms")
        if not answerable:
            if pages or terms or expected_kind.strip() != "unsupported":
                raise GroundTruthValidationError(f"{question_id}: unsupported entries require no pages/terms and kind unsupported")
            normalized_accepted = {normalize_evaluation_text(value) for value in accepted}
            if normalized_accepted and not normalized_accepted.issubset(abstention_terms):
                raise GroundTruthValidationError(f"{question_id}: unsupported accepted answers must describe abstention only")
        questions.append(GroundTruthQuestion(
            id=question_id, question=question.strip(), answerable=answerable,
            accepted_answers=accepted, expected_pages=tuple(pages),
            expected_kind=expected_kind.strip(), expected_terms=terms,
        ))
    if len(questions) < 2 or sum(not item.answerable for item in questions) < 2:
        raise GroundTruthValidationError("Ground truth must include at least two unsupported questions")
    questions.sort(key=lambda item: item.id)
    return GroundTruthSet(tuple(questions), _hash_bytes(raw), GROUND_TRUTH_RELATIVE_PATH)


def _index_section(run: ProcessingRun) -> dict[str, Any]:
    value = (run.configuration_json or {}).get("embedding_indexing")
    return dict(value) if isinstance(value, dict) else {}


def _safe_source_path(document: Document) -> Path:
    candidate = Path(document.storage_path)
    if not candidate.is_absolute():
        candidate = _project_root() / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise GroundTruthValidationError("The persisted source PDF is unavailable")
    return resolved


def build_evaluation_plan(
    *, document_id: uuid.UUID, baseline_processing_run_id: uuid.UUID,
    docling_processing_run_id: uuid.UUID, top_k: int = 5,
    settings: Settings | None = None,
) -> EvaluationPlan:
    """Validate database/index identity without contacting Chroma or Azure."""

    configured = settings or get_settings()
    if top_k != configured.rag_top_k:
        raise EvaluationConflictError(f"Controlled evaluation top_k must match configured retrieval top_k ({configured.rag_top_k})")
    if not configured.rag_model_ready:
        raise ConfigurationError("Azure OpenAI RAG models are not configured: " + ", ".join(configured.azure_openai_missing_settings))
    prompt = load_rag_prompt(configured.rag_prompt_version)
    with SessionLocal() as session:
        document = session.get(Document, document_id)
        baseline = session.get(ProcessingRun, baseline_processing_run_id)
        docling = session.get(ProcessingRun, docling_processing_run_id)
        if document is None:
            raise EvaluationNotFoundError(f"Document not found: {document_id}")
        if baseline is None or docling is None:
            raise EvaluationNotFoundError("One or more processing runs were not found")
        for run, expected_pipeline in ((baseline, PipelineType.BASELINE.value), (docling, PipelineType.DOCLING_STANDARD.value)):
            if run.document_id != document_id:
                raise EvaluationConflictError("A selected processing run belongs to another document")
            if run.pipeline_type != expected_pipeline:
                raise EvaluationConflictError(f"Selected run {run.id} is not pipeline {expected_pipeline}")
            if run.status != ProcessingStatus.COMPLETED.value:
                raise EvaluationConflictError(f"Selected run {run.id} is not completed")
        page_count = document.page_count or 10
        source_sha256 = _hash_bytes(_safe_source_path(document).read_bytes())
        if source_sha256 != document.sha256:
            raise EvaluationConflictError("Persisted source hash does not match the source PDF")
        baseline_duration, docling_duration = baseline.duration_ms, docling.duration_ms
        baseline_section, docling_section = _index_section(baseline), _index_section(docling)
    ground_truth = load_ground_truth(maximum_page=page_count)
    baseline_ready = inspect_run_readiness(baseline_processing_run_id, settings=configured)
    docling_ready = inspect_run_readiness(docling_processing_run_id, settings=configured)
    if not baseline_ready.fully_indexed or not docling_ready.fully_indexed:
        issues = list(baseline_ready.indexing_issues) + list(docling_ready.indexing_issues)
        raise RunNotIndexedError("Selected runs are not fully indexed: " + ", ".join(issues))
    shared_fields = ("deployment_name", "service_returned_model", "vector_dimension", "chroma_distance")
    mismatched = [field for field in shared_fields if baseline_section.get(field) != docling_section.get(field)]
    if mismatched:
        raise EvaluationConflictError("The controlled indexes do not share required settings: " + ", ".join(mismatched))
    configuration = {
        "embedding_deployment": baseline_section.get("deployment_name"),
        "embedding_service_model": baseline_section.get("service_returned_model"),
        "vector_dimension": baseline_section.get("vector_dimension"),
        "chroma_distance": baseline_section.get("chroma_distance"),
        "chat_deployment": configured.azure_openai_chat_deployment,
        "chat_service_model": configured.azure_openai_chat_deployment,
        "chat_api": "Azure OpenAI-compatible Responses API",
        "chat_temperature": configured.chat_temperature,
        "chat_reasoning_effort": configured.chat_reasoning_effort,
        "chat_max_output_tokens": configured.chat_max_output_tokens,
        "structured_output_retry_count": configured.chat_structured_output_retry_count,
        "prompt_version": prompt.version, "prompt_sha256": prompt.sha256,
        "response_schema_version": prompt.schema_version, "top_k": top_k,
        "collections": {"baseline": baseline_ready.expected_collection, "docling_standard": docling_ready.expected_collection},
        "embedding_fingerprints": {"baseline": baseline_ready.embedding_fingerprint, "docling_standard": docling_ready.embedding_fingerprint},
        "indexed_chunk_counts": {"baseline": baseline_ready.indexed_chunk_count, "docling_standard": docling_ready.indexed_chunk_count},
    }
    identity = {
        "source_pdf_sha256": source_sha256, "ground_truth_sha256": ground_truth.sha256,
        "baseline_processing_run_id": str(baseline_processing_run_id),
        "docling_processing_run_id": str(docling_processing_run_id),
        "baseline_collection": baseline_ready.expected_collection,
        "baseline_embedding_fingerprint": baseline_ready.embedding_fingerprint,
        "docling_collection": docling_ready.expected_collection,
        "docling_embedding_fingerprint": docling_ready.embedding_fingerprint,
        "embedding_deployment": configuration["embedding_deployment"],
        "vector_dimension": configuration["vector_dimension"],
        "chat_deployment": configuration["chat_deployment"],
        "prompt_version": prompt.version, "prompt_sha256": prompt.sha256,
        "response_schema_version": prompt.schema_version, "top_k": top_k,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
    }
    processing_metrics = {
        "source_page_count": page_count,
        "baseline_ingestion_duration_ms": baseline_duration,
        "docling_standard_conversion_chunking_duration_ms": docling_duration,
        "baseline_duration_per_page_ms": round(baseline_duration / page_count, 2) if baseline_duration is not None else None,
        "docling_duration_per_page_ms": round(docling_duration / page_count, 2) if docling_duration is not None else None,
        "chunk_provenance_backfill_duration_ms": 466,
        "timing_note": "Existing persisted durations for one synthetic 10-page document; no processing was rerun.",
    }
    return EvaluationPlan(document_id, baseline_processing_run_id, docling_processing_run_id, top_k,
                          ground_truth, source_sha256, _canonical_hash(identity), configuration, processing_metrics)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _limitations(plan: EvaluationPlan) -> list[str]:
    return [
        "One synthetic 10-page document and one evaluation pass; results are descriptive, not a universal parser ranking.",
        f"The Baseline index contains only {plan.configuration['indexed_chunk_counts']['baseline']} broad chunks, so Recall@5 is less discriminating.",
        "Citation integrity validates supplied IDs and structure; it does not prove semantic entailment.",
        "Precise provenance supports verification but does not eliminate hallucination.",
        "Docling heading reconstruction and some vertical table merges were imperfect.",
        "Picture descriptions were generic derived retrieval aids, not verbatim source evidence.",
        "Header repetition was configured but not exercised because production table chunks fit under 800 tokens.",
        "Docling CPU conversion was slow for this quality profile.",
        "Granite-Docling-258M was intentionally not evaluated.",
    ]


def _base_summary(plan: EvaluationPlan, *, status: str) -> dict[str, Any]:
    now = _utcnow().isoformat()
    return {
        "status": status, "evaluation_fingerprint": plan.fingerprint,
        "evaluation_version": EVALUATION_VERSION, "metric_definition_version": METRIC_DEFINITION_VERSION,
        "document_id": str(plan.document_id),
        "baseline_processing_run_id": str(plan.baseline_processing_run_id),
        "docling_processing_run_id": str(plan.docling_processing_run_id),
        "ground_truth_path": plan.ground_truth.relative_path,
        "ground_truth_sha256": plan.ground_truth.sha256, "source_sha256": plan.source_sha256,
        "question_count": len(plan.ground_truth.questions),
        "answerable_count": plan.ground_truth.answerable_count,
        "unsupported_count": plan.ground_truth.unsupported_count,
        "total_case_count": plan.total_cases, "completed_case_count": 0,
        "progress_percent": 0, "current_question_id": None, "queued_at": now,
        "started_at": None, "completed_at": None, "last_updated_at": now,
        "duration_ms": None, "configuration": plan.configuration,
        "processing_metrics": plan.processing_metrics, "metric_definitions": METRIC_DEFINITIONS,
        "pipeline_summaries": {}, "category_summaries": [],
        "failure_summary": {"case_failures": 0, "provider_failures": 0, "details": []},
        "limitations": _limitations(plan), "export_urls": {},
    }


def create_or_reuse_evaluation(plan: EvaluationPlan, *, force_new: bool = False) -> tuple[uuid.UUID, bool]:
    """Create one queued row or reuse an identical completed evaluation."""

    with SessionLocal.begin() as session:
        existing = list(session.scalars(select(EvaluationRun).where(EvaluationRun.document_id == plan.document_id)
                                        .order_by(EvaluationRun.created_at.desc()).with_for_update()))
        if not force_new:
            reused = next((run for run in existing
                           if (run.summary_json or {}).get("evaluation_fingerprint") == plan.fingerprint
                           and (run.summary_json or {}).get("status") in TERMINAL_REUSABLE_STATUSES), None)
            if reused is not None:
                return reused.id, True
        active = [run for run in existing if (run.summary_json or {}).get("status") in ACTIVE_STATUSES]
        if active:
            raise EvaluationConflictError(f"Evaluation {active[0].id} is already active; wait for it to finish")
        evaluation = EvaluationRun(document_id=plan.document_id, results_json=[], summary_json=_base_summary(plan, status="queued"))
        session.add(evaluation)
        session.flush()
        return evaluation.id, False


def _persist_progress(evaluation_id: uuid.UUID, *, results: list[dict[str, Any]], status: str,
                      current_question_id: str | None, started_at: datetime,
                      extra: dict[str, Any] | None = None) -> None:
    now = _utcnow()
    with SessionLocal.begin() as session:
        evaluation = session.get(EvaluationRun, evaluation_id, with_for_update=True)
        if evaluation is None:
            raise EvaluationNotFoundError(str(evaluation_id))
        summary = dict(evaluation.summary_json or {})
        total = int(summary.get("total_case_count") or 0)
        summary.update({
            "status": status, "started_at": summary.get("started_at") or started_at.isoformat(),
            "last_updated_at": now.isoformat(), "current_question_id": current_question_id,
            "completed_case_count": len(results),
            "progress_percent": round((len(results) / total) * 100, 2) if total else 0,
            "elapsed_duration_ms": max(0, round((now - started_at).total_seconds() * 1000)),
        })
        if extra:
            summary.update(extra)
        evaluation.results_json = list(results)
        evaluation.summary_json = summary


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _all_strings(child)


def _source_search_text(chunk: Chunk) -> str:
    metadata = chunk.chunk_metadata or {}
    derived: list[str] = []
    for component in metadata.get("picture_components") or []:
        if isinstance(component, dict):
            description = component.get("generated_description")
            if isinstance(description, dict) and isinstance(description.get("text"), str):
                derived.append(description["text"])
    parts = [chunk.raw_text, chunk.embedding_text, *_all_strings(chunk.captions or []), *_all_strings(metadata)]
    cleaned: list[str] = []
    for part in parts:
        for description in derived:
            part = part.replace(description, " ")
        cleaned.append(part)
    return "\n".join(cleaned)


def _page_overlap(chunk: Chunk, expected_pages: tuple[int, ...]) -> tuple[bool, list[int]]:
    if not expected_pages:
        return False, []
    precise_pages = sorted({record.page_no for record in chunk.provenance_records})
    if precise_pages:
        pages = precise_pages
    elif chunk.page_start is not None and chunk.page_end is not None:
        pages = list(range(chunk.page_start, chunk.page_end + 1))
    else:
        pages = []
    return bool(set(pages).intersection(expected_pages)), pages


def _score_retrieval_hits(response: RagResponse, question: GroundTruthQuestion) -> tuple[list[dict[str, Any]], int | None]:
    chunk_ids = [hit.chunk_id for hit in response.retrieval_hits]
    with SessionLocal() as session:
        chunks = list(session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))
                                      .options(selectinload(Chunk.provenance_records)))) if chunk_ids else []
    by_id = {chunk.id: chunk for chunk in chunks}
    normalized_terms = [normalize_evaluation_text(term) for term in question.expected_terms]
    details: list[dict[str, Any]] = []
    first_rank: int | None = None
    for hit in response.retrieval_hits:
        chunk = by_id.get(hit.chunk_id)
        if chunk is None:
            continue
        overlap, pages = _page_overlap(chunk, question.expected_pages)
        searchable = normalize_evaluation_text(_source_search_text(chunk))
        coverage = [{"term": original, "normalized_term": normalized,
                     "present": _contains_normalized_phrase(searchable, normalized)}
                    for original, normalized in zip(question.expected_terms, normalized_terms)]
        relevant = bool(question.answerable and overlap and coverage and all(item["present"] for item in coverage))
        if relevant and first_rank is None:
            first_rank = hit.rank
        details.append({
            "rank": hit.rank, "evidence_id": hit.evidence_id, "chunk_id": str(hit.chunk_id),
            "distance": hit.distance, "page_start": hit.page_start, "page_end": hit.page_end,
            "precise_provenance_pages": list(hit.available_provenance_pages),
            "resolved_pages": pages, "kind": hit.kind, "relevant": relevant,
            "expected_term_coverage": coverage, "page_overlap": overlap,
            "source_classification": hit.source_classification,
        })
    return details, first_rank


def _answer_match(response: RagResponse, question: GroundTruthQuestion) -> tuple[bool | None, str | None, str | None, str]:
    normalized_answer = normalize_evaluation_text(response.answer)
    if not question.answerable:
        return None, None, None, normalized_answer
    if response.insufficient_evidence:
        return False, None, "abstained", normalized_answer
    answer_tokens = normalized_answer.split()
    answer_token_set = set(answer_tokens)
    for accepted in question.accepted_answers:
        accepted_tokens = normalize_evaluation_text(accepted).split()
        if accepted_tokens and any(answer_tokens[index:index + len(accepted_tokens)] == accepted_tokens
                                   for index in range(0, len(answer_tokens) - len(accepted_tokens) + 1)):
            return True, accepted, "normalized_phrase", normalized_answer
        meaningful = [token for token in accepted_tokens if token not in STOPWORDS]
        if meaningful and all(token in answer_token_set for token in meaningful):
            return True, accepted, "meaningful_token_coverage", normalized_answer
    return False, None, "no_match", normalized_answer


def _resolved_pages(citation: dict[str, Any]) -> list[int]:
    precise = citation.get("available_provenance_pages") or []
    if precise:
        return sorted({int(value) for value in precise})
    start, end = citation.get("page_start"), citation.get("page_end")
    return list(range(start, end + 1)) if isinstance(start, int) and isinstance(end, int) else []


def _citation_scores(response: RagResponse, question: GroundTruthQuestion) -> tuple[bool | None, bool | None, bool | None, list[int], list[str], list[dict[str, Any]]]:
    if not question.answerable or response.insufficient_evidence:
        return None, None, None, [], [], []
    integrity = response.citation_validation_status == "valid"
    citations = [item.model_dump(mode="json") for item in response.resolved_citations]
    by_id = {item["evidence_id"]: item for item in citations}
    claim_page_results: list[dict[str, Any]] = []
    all_claims_correct = bool(response.claims)
    for index, claim in enumerate(response.claims):
        valid_ids = [value for value in claim.citation_ids if value in by_id]
        expected_page_ids = [value for value in valid_ids
                             if set(_resolved_pages(by_id[value])).intersection(question.expected_pages)]
        correct = bool(valid_ids and expected_page_ids)
        all_claims_correct = all_claims_correct and correct
        claim_page_results.append({
            "claim_index": index, "claim_text": claim.text,
            "citation_ids": list(claim.citation_ids), "valid_citation_ids": valid_ids,
            "expected_page_citation_ids": expected_page_ids, "page_correct": correct,
        })
    cited_pages = sorted({page for item in citations for page in _resolved_pages(item)})
    cited_chunks = list(dict.fromkeys(str(item["chunk_id"]) for item in citations))
    precise = any(bool(item.get("precise_provenance_available")) for item in citations)
    return integrity, all_claims_correct, precise, cited_pages, cited_chunks, claim_page_results


def _latest_query_run(*, document_id: uuid.UUID, pipeline: str, question: str, since: datetime) -> QueryRun | None:
    with SessionLocal() as session:
        run = session.scalar(select(QueryRun).where(
            QueryRun.document_id == document_id, QueryRun.pipeline_type == pipeline,
            QueryRun.question == question, QueryRun.created_at >= since,
        ).order_by(QueryRun.created_at.desc()).limit(1))
        if run is not None:
            session.expunge(run)
        return run


def _safe_failure_category(exc: Exception) -> str:
    if isinstance(exc, GenerationValidationError):
        return "generation_validation_failure"
    if isinstance(exc, RunNotIndexedError):
        return "run_not_indexed"
    if isinstance(exc, ConfigurationError):
        return "model_configuration_unavailable"
    if isinstance(exc, ApplicationError):
        return "provider_or_service_failure"
    return "unexpected_evaluation_failure"


def _failure_case(evaluation_id: uuid.UUID, question: GroundTruthQuestion, pipeline: str,
                  top_k: int, exc: Exception, query_run: QueryRun | None) -> dict[str, Any]:
    metadata = dict(query_run.answer_json or {}) if query_run is not None else {}
    attempts = list(metadata.get("validation_attempts") or [])
    usage = dict(metadata.get("token_usage") or {})
    return {
        "evaluation_id": str(evaluation_id), "question_id": question.id, "pipeline": pipeline,
        "pipeline_label": PIPELINE_LABELS[pipeline], "question": question.question,
        "answerable": question.answerable, "expected_kind": question.expected_kind,
        "expected_pages": list(question.expected_pages), "expected_terms": list(question.expected_terms),
        "accepted_answers": list(question.accepted_answers), "requested_top_k": top_k,
        "actual_hit_count": int(metadata.get("actual_hit_count") or 0),
        "query_run_id": str(query_run.id) if query_run else None, "ranked_retrieval": [],
        "first_relevant_rank": None, "reciprocal_rank": 0.0 if question.answerable else None,
        "recall_at_1": False if question.answerable else None,
        "recall_at_3": False if question.answerable else None,
        "recall_at_5": False if question.answerable else None,
        "ingestion_or_representation_miss": bool(question.answerable),
        "answer": query_run.answer_text if query_run else None,
        "normalized_answer": normalize_evaluation_text(query_run.answer_text) if query_run and query_run.answer_text else None,
        "insufficient_evidence": query_run.insufficient_evidence if query_run else None,
        "normalized_answer_match": False if question.answerable else None,
        "matched_accepted_answer": None, "answer_match_method": "case_failure",
        "claim_count": len(metadata.get("claims") or []), "claims": list(metadata.get("claims") or []),
        "citation_ids": list(metadata.get("citation_ids") or []), "citation_integrity_valid": None,
        "citation_page_correct": None, "precise_provenance_available": None,
        "citation_resolution": [], "claim_page_resolution": [], "cited_pages": [],
        "cited_chunk_ids": [], "structural_abstention_correct": False if not question.answerable else None,
        "answerability_decision_correct": False,
        "retrieval_duration_ms": query_run.retrieval_duration_ms if query_run else None,
        "generation_duration_ms": query_run.generation_duration_ms if query_run else None,
        "total_duration_ms": None, "evidence_token_count": metadata.get("total_evidence_tokens"),
        "model_input_tokens": usage.get("input_tokens"), "model_output_tokens": usage.get("output_tokens"),
        "model_total_tokens": usage.get("total_tokens"), "validation_attempt_count": len(attempts),
        "provider": metadata.get("provider"),
        "deployment": metadata.get("model_deployment"),
        "service_model": metadata.get("service_returned_model"),
        "failure_category": _safe_failure_category(exc),
        "safe_error": "The case could not be completed; inspect server logs with its evaluation and question IDs.",
        "result_status": "error",
    }


def _score_case(evaluation_id: uuid.UUID, question: GroundTruthQuestion, pipeline: str,
                response: RagResponse) -> dict[str, Any]:
    ranked, first_rank = _score_retrieval_hits(response, question)
    answer_match, matched, match_method, normalized_answer = _answer_match(response, question)
    integrity, page_correct, precise, cited_pages, cited_chunks, claim_pages = _citation_scores(response, question)
    structural_abstention = (bool(response.answer.strip()) and response.insufficient_evidence and not response.claims
                             if not question.answerable else None)
    decision_correct = not response.insufficient_evidence if question.answerable else bool(structural_abstention)
    passed = bool(answer_match and integrity and page_correct and decision_correct) if question.answerable else bool(structural_abstention)
    return {
        "evaluation_id": str(evaluation_id), "question_id": question.id, "pipeline": pipeline,
        "pipeline_label": PIPELINE_LABELS[pipeline], "question": question.question,
        "answerable": question.answerable, "expected_kind": question.expected_kind,
        "expected_pages": list(question.expected_pages), "expected_terms": list(question.expected_terms),
        "accepted_answers": list(question.accepted_answers), "requested_top_k": response.requested_top_k,
        "actual_hit_count": response.actual_hit_count, "query_run_id": str(response.query_run_id),
        "ranked_retrieval": ranked, "first_relevant_rank": first_rank if question.answerable else None,
        "reciprocal_rank": (1.0 / first_rank if first_rank else 0.0) if question.answerable else None,
        "recall_at_1": bool(first_rank and first_rank <= 1) if question.answerable else None,
        "recall_at_3": bool(first_rank and first_rank <= 3) if question.answerable else None,
        "recall_at_5": bool(first_rank and first_rank <= 5) if question.answerable else None,
        "ingestion_or_representation_miss": bool(question.answerable and first_rank is None),
        "answer": response.answer, "normalized_answer": normalized_answer,
        "insufficient_evidence": response.insufficient_evidence,
        "normalized_answer_match": answer_match, "matched_accepted_answer": matched,
        "answer_match_method": match_method, "claim_count": len(response.claims),
        "claims": [claim.model_dump(mode="json") for claim in response.claims],
        "citation_ids": list(response.citation_ids), "citation_integrity_valid": integrity,
        "citation_page_correct": page_correct, "precise_provenance_available": precise,
        "citation_resolution": [item.model_dump(mode="json") for item in response.resolved_citations],
        "claim_page_resolution": claim_pages, "cited_pages": cited_pages,
        "cited_chunk_ids": cited_chunks, "structural_abstention_correct": structural_abstention,
        "answerability_decision_correct": decision_correct,
        "retrieval_duration_ms": response.retrieval_duration_ms,
        "generation_duration_ms": response.generation_duration_ms,
        "total_duration_ms": response.total_duration_ms,
        "evidence_token_count": response.total_evidence_tokens,
        "model_input_tokens": response.usage.input_tokens,
        "model_output_tokens": response.usage.output_tokens,
        "model_total_tokens": response.usage.total_tokens,
        "validation_attempt_count": response.validation_attempt_count,
        "provider": response.provider, "deployment": response.deployment,
        "service_model": response.service_model, "response_id": response.response_id,
        "prompt_version": response.prompt_version, "prompt_sha256": response.prompt_hash,
        "response_schema_version": response.schema_version,
        "failure_category": None, "result_status": "pass" if passed else "fail",
    }


def _ratio(values: list[dict[str, Any]], field: str) -> dict[str, Any]:
    applicable = [item for item in values if item.get(field) is not None]
    numerator, denominator = sum(bool(item.get(field)) for item in applicable), len(applicable)
    return {"value": round(numerator / denominator, 6) if denominator else None,
            "numerator": numerator, "denominator": denominator}


def _numeric_summary(values: Iterable[int | float | None]) -> dict[str, Any]:
    present = sorted(float(value) for value in values if value is not None)
    if not present:
        return {"mean": None, "median": None, "p95": None, "count": 0}
    p95_index = max(0, math.ceil(0.95 * len(present)) - 1)
    return {"mean": round(statistics.fmean(present), 2), "median": round(statistics.median(present), 2),
            "p95": round(present[p95_index], 2), "count": len(present)}


def _pipeline_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in cases if item["answerable"]]
    unsupported = [item for item in cases if not item["answerable"]]
    table_cases = [item for item in answerable if item["expected_kind"] == "table"]
    supported_answers = [item for item in answerable
                         if item.get("insufficient_evidence") is False and item.get("result_status") != "error"]
    mrr_values = [float(item["reciprocal_rank"] or 0) for item in answerable]
    repair_count = sum(max(0, int(item.get("validation_attempt_count") or 0) - 1) for item in cases)
    non_abstentions = sum(item.get("insufficient_evidence") is False for item in answerable)
    return {
        "question_count": len(cases), "answerable_count": len(answerable), "unsupported_count": len(unsupported),
        "retrieval": {
            "recall_at_1": _ratio(answerable, "recall_at_1"), "recall_at_3": _ratio(answerable, "recall_at_3"),
            "recall_at_5": _ratio(answerable, "recall_at_5"),
            "mrr": {"value": round(statistics.fmean(mrr_values), 6) if mrr_values else None,
                    "denominator": len(mrr_values)},
            "representation_miss_count": sum(bool(item.get("ingestion_or_representation_miss")) for item in answerable),
            "mean_actual_hit_count": round(statistics.fmean(float(item["actual_hit_count"]) for item in cases), 2) if cases else None,
        },
        "answer": {"normalized_answer_match_rate": _ratio(answerable, "normalized_answer_match"),
                   "table_question_accuracy": _ratio(table_cases, "normalized_answer_match")},
        "citation": {"citation_integrity_rate": _ratio(supported_answers, "citation_integrity_valid"),
                     "citation_page_accuracy": _ratio(supported_answers, "citation_page_correct"),
                     "precise_provenance_availability_rate": _ratio(supported_answers, "precise_provenance_available")},
        "abstention": {"unsupported_abstention_accuracy": _ratio(unsupported, "structural_abstention_correct"),
                       "supported_non_abstention_rate": {"value": round(non_abstentions / len(answerable), 6) if answerable else None,
                                                          "numerator": non_abstentions, "denominator": len(answerable)},
                       "answerability_decision_accuracy": _ratio(cases, "answerability_decision_correct")},
        "latency_ms": {"retrieval": _numeric_summary(item.get("retrieval_duration_ms") for item in cases),
                       "generation": _numeric_summary(item.get("generation_duration_ms") for item in cases),
                       "total": _numeric_summary(item.get("total_duration_ms") for item in cases)},
        "tokens": {"total_chat_input_tokens": sum(int(item.get("model_input_tokens") or 0) for item in cases),
                   "total_chat_output_tokens": sum(int(item.get("model_output_tokens") or 0) for item in cases),
                   "total_chat_tokens": sum(int(item.get("model_total_tokens") or 0) for item in cases),
                   "mean_evidence_tokens_per_question": round(statistics.fmean(float(item.get("evidence_token_count") or 0) for item in cases), 2) if cases else None,
                   "validation_repair_attempt_count": repair_count},
        "provider_failure_count": sum(item.get("failure_category") in {"provider_or_service_failure", "model_configuration_unavailable"} for item in cases),
        "case_failure_count": sum(item.get("result_status") == "error" for item in cases),
    }


def _category_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pipeline in PIPELINE_LABELS:
        for category in sorted({item["expected_kind"] for item in results}):
            cases = [item for item in results if item["pipeline"] == pipeline and item["expected_kind"] == category]
            output.append({"pipeline": pipeline, "pipeline_label": PIPELINE_LABELS[pipeline],
                           "expected_kind": category, "question_count": len(cases), **_pipeline_summary(cases)})
    return output


def _summarize(plan: EvaluationPlan, results: list[dict[str, Any]]) -> dict[str, Any]:
    pipelines = {pipeline: _pipeline_summary([item for item in results if item["pipeline"] == pipeline])
                 for pipeline in PIPELINE_LABELS}
    failures = [item for item in results if item.get("result_status") == "error"]
    provider_failures = [item for item in failures if item.get("failure_category") in {"provider_or_service_failure", "model_configuration_unavailable"}]
    return {
        "pipeline_summaries": pipelines, "category_summaries": _category_summaries(results),
        "failure_summary": {
            "case_failures": len(failures), "provider_failures": len(provider_failures),
            "failure_rate": round(len(failures) / len(results), 6) if results else 0,
            "details": [{"question_id": item["question_id"], "pipeline": item["pipeline"],
                         "category": item.get("failure_category")} for item in failures],
        },
        "processing_metrics": plan.processing_metrics, "limitations": _limitations(plan),
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def evaluation_csv(evaluation_id: uuid.UUID) -> str:
    evaluation = get_evaluation_record(evaluation_id)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for result in list(evaluation.results_json or []):
        writer.writerow({key: _csv_value(result.get(key)) for key in CSV_COLUMNS})
    return output.getvalue()


def evaluation_json_payload(evaluation_id: uuid.UUID) -> dict[str, Any]:
    evaluation = get_evaluation_record(evaluation_id)
    summary = dict(evaluation.summary_json or {})
    return {
        "evaluation_metadata": {"evaluation_id": str(evaluation.id), "document_id": str(evaluation.document_id),
                                "status": summary.get("status"), "created_at": evaluation.created_at.isoformat(),
                                "started_at": summary.get("started_at"), "completed_at": summary.get("completed_at"),
                                "duration_ms": summary.get("duration_ms"),
                                "evaluation_version": summary.get("evaluation_version"),
                                "metric_definition_version": summary.get("metric_definition_version"),
                                "evaluation_fingerprint": summary.get("evaluation_fingerprint")},
        "metric_definitions": summary.get("metric_definitions", METRIC_DEFINITIONS),
        "ground_truth_sha256": summary.get("ground_truth_sha256"), "source_sha256": summary.get("source_sha256"),
        "configuration": summary.get("configuration", {}),
        "question_counts": {"total": summary.get("question_count"), "answerable": summary.get("answerable_count"),
                            "unsupported": summary.get("unsupported_count"), "pipeline_cases": summary.get("total_case_count")},
        "overall_summary": {
            "status": summary.get("status"),
            "completed_case_count": summary.get("completed_case_count"),
            "total_case_count": summary.get("total_case_count"),
            "failure_summary": summary.get("failure_summary", {}),
        },
        "pipeline_summaries": summary.get("pipeline_summaries", {}),
        "category_summaries": summary.get("category_summaries", []),
        "per_question_results": list(evaluation.results_json or []),
        "processing_metrics": summary.get("processing_metrics", {}),
        "failures": summary.get("failure_summary", {}), "limitations": summary.get("limitations", []),
    }


def _write_exports(evaluation_id: uuid.UUID, settings: Settings) -> None:
    output_dir = (settings.artifact_root / "evaluations" / str(evaluation_id)).resolve()
    artifact_root = settings.artifact_root.resolve()
    if output_dir != artifact_root and artifact_root not in output_dir.parents:
        raise RuntimeError("Evaluation export directory is outside the artifact root")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"evaluation-{evaluation_id}.csv").write_text(evaluation_csv(evaluation_id), encoding="utf-8", newline="")
    (output_dir / f"evaluation-{evaluation_id}.json").write_text(
        json.dumps(evaluation_json_payload(evaluation_id), indent=2, ensure_ascii=False), encoding="utf-8")


def execute_evaluation(evaluation_id: uuid.UUID, *, plan: EvaluationPlan | None = None,
                       settings: Settings | None = None,
                       progress_callback: Callable[[dict[str, Any]], None] | None = None) -> EvaluationRun:
    """Execute all baseline/Docling pairs sequentially exactly once."""

    configured = settings or get_settings()
    evaluation = get_evaluation_record(evaluation_id)
    initial = dict(evaluation.summary_json or {})
    if initial.get("status") in TERMINAL_REUSABLE_STATUSES:
        return evaluation
    if plan is None:
        plan = build_evaluation_plan(
            document_id=evaluation.document_id,
            baseline_processing_run_id=uuid.UUID(initial["baseline_processing_run_id"]),
            docling_processing_run_id=uuid.UUID(initial["docling_processing_run_id"]),
            top_k=int(initial["configuration"]["top_k"]), settings=configured)
    started_at = _utcnow()
    results: list[dict[str, Any]] = list(evaluation.results_json or [])
    _persist_progress(evaluation_id, results=results, status="running", current_question_id=None, started_at=started_at)
    run_by_pipeline = {PipelineType.BASELINE.value: plan.baseline_processing_run_id,
                       PipelineType.DOCLING_STANDARD.value: plan.docling_processing_run_id}
    completed_keys = {(item["pipeline"], item["question_id"]) for item in results}
    broad_provider_failure = False
    try:
        for question in plan.ground_truth.questions:
            for pipeline in (PipelineType.BASELINE.value, PipelineType.DOCLING_STANDARD.value):
                if (pipeline, question.id) in completed_keys:
                    continue
                case_started = _utcnow()
                try:
                    response = answer_question(run_by_pipeline[pipeline], question.question,
                                               top_k=plan.top_k, settings=configured)
                    case = _score_case(evaluation_id, question, pipeline, response)
                except Exception as exc:
                    query_run = _latest_query_run(document_id=plan.document_id, pipeline=pipeline,
                                                  question=question.question,
                                                  since=case_started - timedelta(seconds=1))
                    case = _failure_case(evaluation_id, question, pipeline, plan.top_k, exc, query_run)
                    logger.exception("evaluation_id=%s question_id=%s pipeline=%s case_failed",
                                     evaluation_id, question.id, pipeline)
                results.append(case)
                _persist_progress(evaluation_id, results=results, status="running",
                                  current_question_id=question.id, started_at=started_at)
                if progress_callback:
                    progress_callback({"evaluation_id": str(evaluation_id), "question_id": question.id,
                                       "pipeline": pipeline, "completed_case_count": len(results),
                                       "total_case_count": plan.total_cases, "result_status": case["result_status"]})
                provider_failures = sum(
                    item.get("failure_category") in {
                        "provider_or_service_failure", "model_configuration_unavailable"
                    }
                    for item in results
                )
                if len(results) >= 4 and provider_failures / len(results) > 0.25:
                    broad_provider_failure = True
                    logger.error(
                        "evaluation_id=%s aborted_after=%s provider_failures=%s",
                        evaluation_id, len(results), provider_failures,
                    )
                    break
            if broad_provider_failure:
                break
        observed_models = sorted({str(item["service_model"]) for item in results if item.get("service_model")})
        observed_deployments = sorted({str(item["deployment"]) for item in results if item.get("deployment")})
        plan.configuration["observed_chat_service_models"] = observed_models
        plan.configuration["observed_chat_deployments"] = observed_deployments
        if len(observed_models) == 1:
            plan.configuration["chat_service_model"] = observed_models[0]
        aggregates = _summarize(plan, results)
        failure_rate = aggregates["failure_summary"]["failure_rate"]
        final_status = "failed" if broad_provider_failure or failure_rate > 0.25 else (
            "completed_with_failures" if aggregates["failure_summary"]["case_failures"] else "completed")
        completed_at = _utcnow()
        elapsed = max(0, round((completed_at - started_at).total_seconds() * 1000))
        extra = {**aggregates, "configuration": plan.configuration,
                 "status": final_status, "completed_at": completed_at.isoformat(),
                 "current_question_id": None,
                 "progress_percent": round(len(results) / plan.total_cases * 100, 2),
                 "duration_ms": elapsed,
                 "elapsed_duration_ms": elapsed,
                 "export_urls": ({"csv": f"/evaluation/{evaluation_id}/export.csv",
                                  "json": f"/evaluation/{evaluation_id}/export.json"}
                                 if final_status in TERMINAL_REUSABLE_STATUSES else {}),
                 "safe_error": ({
                     "category": "broad_provider_failure",
                     "message": "Evaluation stopped after provider failures exceeded the allowed threshold.",
                 } if broad_provider_failure else None)}
        _persist_progress(evaluation_id, results=results, status=final_status, current_question_id=None,
                          started_at=started_at, extra=extra)
        if final_status in TERMINAL_REUSABLE_STATUSES:
            _write_exports(evaluation_id, configured)
        return get_evaluation_record(evaluation_id)
    except Exception as exc:
        completed_at = _utcnow()
        _persist_progress(evaluation_id, results=results, status="failed", current_question_id=None,
                          started_at=started_at, extra={
                              "completed_at": completed_at.isoformat(),
                              "duration_ms": max(0, round((completed_at - started_at).total_seconds() * 1000)),
                              "safe_error": {"category": "evaluation_execution_failure",
                                             "message": "Evaluation stopped because the runner failed unexpectedly.",
                                             "exception_type": type(exc).__name__}})
        raise


def get_evaluation_record(evaluation_id: uuid.UUID) -> EvaluationRun:
    with SessionLocal() as session:
        evaluation = session.get(EvaluationRun, evaluation_id)
        if evaluation is None:
            raise EvaluationNotFoundError(str(evaluation_id))
        session.expunge(evaluation)
        return evaluation


def evaluation_view(evaluation_id: uuid.UUID, *, include_results: bool = False,
                    reused: bool = False) -> dict[str, Any]:
    evaluation = get_evaluation_record(evaluation_id)
    summary = dict(evaluation.summary_json or {})
    return {
        "id": str(evaluation.id), "evaluation_id": str(evaluation.id),
        "document_id": str(evaluation.document_id), "status": summary.get("status", "unknown"),
        "reused": reused, "progress_percent": summary.get("progress_percent", 0),
        "current_question_id": summary.get("current_question_id"),
        "completed_case_count": summary.get("completed_case_count", 0),
        "total_case_count": summary.get("total_case_count", 0),
        "question_count": summary.get("question_count", 0),
        "answerable_count": summary.get("answerable_count", 0),
        "unsupported_count": summary.get("unsupported_count", 0),
        "started_at": summary.get("started_at"), "completed_at": summary.get("completed_at"),
        "elapsed_duration_ms": summary.get("elapsed_duration_ms"), "duration_ms": summary.get("duration_ms"),
        "configuration": summary.get("configuration", {}),
        "evaluation_version": summary.get("evaluation_version"),
        "metric_definition_version": summary.get("metric_definition_version"),
        "ground_truth_sha256": summary.get("ground_truth_sha256"), "source_sha256": summary.get("source_sha256"),
        "pipeline_summaries": summary.get("pipeline_summaries", {}),
        "category_summaries": summary.get("category_summaries", []),
        "failure_summary": summary.get("failure_summary", {}),
        "processing_metrics": summary.get("processing_metrics", {}),
        "limitations": summary.get("limitations", []), "export_urls": summary.get("export_urls", {}),
        "safe_error": summary.get("safe_error"),
        "results": list(evaluation.results_json or []) if include_results else None,
        "created_at": evaluation.created_at.isoformat(),
    }


def recover_stale_evaluations(settings: Settings | None = None, *, now: datetime | None = None) -> tuple[uuid.UUID, ...]:
    """Mark only obviously stale active evaluations interrupted; never resume model calls."""

    configured = settings or get_settings()
    recovered_at = now or _utcnow()
    cutoff = recovered_at - timedelta(minutes=configured.processing_stale_after_minutes)
    recovered: list[uuid.UUID] = []
    with SessionLocal.begin() as session:
        evaluations = list(session.scalars(select(EvaluationRun).with_for_update(skip_locked=True)))
        for evaluation in evaluations:
            summary = dict(evaluation.summary_json or {})
            if summary.get("status") not in ACTIVE_STATUSES:
                continue
            raw_updated = summary.get("last_updated_at") or summary.get("started_at")
            try:
                updated = datetime.fromisoformat(raw_updated) if isinstance(raw_updated, str) else None
            except ValueError:
                updated = None
            if updated is not None and updated > cutoff:
                continue
            summary.update({"status": "interrupted", "completed_at": recovered_at.isoformat(),
                            "last_updated_at": recovered_at.isoformat(), "current_question_id": None,
                            "safe_error": {"category": "evaluation_interrupted",
                                           "message": "A prior application process stopped before evaluation completed."}})
            evaluation.summary_json = summary
            recovered.append(evaluation.id)
    return tuple(recovered)


class EvaluationExecutor:
    """One bounded in-process evaluation thread; cases remain sequential."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._active_evaluation_id: uuid.UUID | None = None

    def start(self, evaluation_id: uuid.UUID, plan: EvaluationPlan) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._active_evaluation_id = evaluation_id
            self._thread = threading.Thread(target=self._run, args=(evaluation_id, plan),
                                            name="parse-before-you-prompt-evaluation", daemon=True)
            self._thread.start()
            return True

    def _run(self, evaluation_id: uuid.UUID, plan: EvaluationPlan) -> None:
        try:
            execute_evaluation(evaluation_id, plan=plan, settings=self._settings)
        except Exception:
            logger.exception("evaluation_id=%s background_runner_failed", evaluation_id)
        finally:
            with self._lock:
                self._active_evaluation_id = None

    @property
    def active_evaluation_id(self) -> uuid.UUID | None:
        with self._lock:
            return self._active_evaluation_id


_executor_lock = threading.Lock()
_executor: EvaluationExecutor | None = None


def get_evaluation_executor(settings: Settings | None = None) -> EvaluationExecutor:
    global _executor
    configured = settings or get_settings()
    with _executor_lock:
        if _executor is None:
            _executor = EvaluationExecutor(configured)
        return _executor
