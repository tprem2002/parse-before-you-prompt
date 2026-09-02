"""Conventional RAG orchestration, persistence, and citation resolution."""

from __future__ import annotations

import uuid
from time import perf_counter

from app.core.config import Settings, get_settings
from app.core.errors import ApplicationError, ConfigurationError
from app.db.models import QueryRun, RetrievalHit
from app.db.session import SessionLocal
from app.prompts import PromptDefinition, load_rag_prompt
from app.providers.chat.azure_openai import get_azure_openai_chat_provider
from app.providers.chat.base import (
    ChatAnswerResult,
    ChatOutputValidationError,
    ChatProviderError,
    EvidenceForModel,
)
from app.schemas.rag import (
    AnswerResponse,
    EvidenceImageEndpoint,
    RagResponse,
    ResolvedCitationResponse,
    RetrievalEvidenceResponse,
    TokenUsageResponse,
)
from app.services.citation_validator import CitationValidationResult, validate_citations
from app.services.retrieval_service import (
    DERIVED_VISUAL_DESCRIPTION_LABEL,
    Evidence,
    RetrievalResult,
    RunNotIndexedError,
    inspect_run_readiness,
    retrieve,
)


class GenerationValidationError(ApplicationError):
    """Structured output remained invalid after the configured repair retry."""

    def __init__(self, message: str, *, query_run_id: uuid.UUID) -> None:
        self.query_run_id = query_run_id
        super().__init__(message)


def _evidence_class(value: str) -> str:
    return {
        "source": "direct source evidence",
        "derived": "derived evidence",
        "mixed": "mixed source and derived evidence",
    }.get(value, "unknown evidence classification")


def _model_evidence(evidence: Evidence) -> EvidenceForModel:
    contextualized_text = evidence.embedding_text
    if (
        evidence.derived_visual_description
        and DERIVED_VISUAL_DESCRIPTION_LABEL not in contextualized_text
    ):
        contextualized_text = (
            f"{contextualized_text}\n\n{DERIVED_VISUAL_DESCRIPTION_LABEL}:\n"
            f"{evidence.derived_visual_description}"
        )
    return EvidenceForModel(
        evidence_id=evidence.evidence_id,
        kind=evidence.kind,
        evidence_class=_evidence_class(evidence.source_classification),
        contextualized_text=contextualized_text,
    )


def _evidence_image(evidence: Evidence) -> EvidenceImageEndpoint | None:
    if not evidence.evidence_overlay_available:
        return None
    return EvidenceImageEndpoint(
        path=f"/chunks/{evidence.chunk_id}/evidence-image",
        available_pages=list(evidence.evidence_image_pages),
        cached_overlay_available=evidence.cached_overlay_available,
    )


def _retrieval_response(evidence: Evidence) -> RetrievalEvidenceResponse:
    return RetrievalEvidenceResponse(
        evidence_id=evidence.evidence_id,
        rank=evidence.rank,
        chunk_id=evidence.chunk_id,
        kind=evidence.kind,
        distance=evidence.distance,
        page_start=evidence.page_start,
        page_end=evidence.page_end,
        section_path=list(evidence.section_path),
        source_captions=list(evidence.source_captions),
        source_classification=evidence.source_classification,
        precise_provenance_available=evidence.precise_provenance_available,
        provenance_region_count=len(evidence.provenance_records),
        available_provenance_pages=sorted(
            {record.page_no for record in evidence.provenance_records}
        ),
        table_ref=evidence.table_ref,
        picture_ref=evidence.picture_ref,
        evidence_image=_evidence_image(evidence),
        contextualized_token_count=evidence.contextualized_token_count,
    )


def _resolved_citation(evidence: Evidence) -> ResolvedCitationResponse:
    base = _retrieval_response(evidence).model_dump()
    return ResolvedCitationResponse(**base, doc_item_refs=list(evidence.doc_item_refs))


def _prompt_metadata(prompt: PromptDefinition) -> dict[str, object]:
    return {
        "prompt_version": prompt.version,
        "prompt_hash": prompt.sha256,
        "schema_version": prompt.schema_version,
    }


def _retrieval_metadata(result: RetrievalResult, prompt: PromptDefinition) -> dict[str, object]:
    return {
        "status": "retrieval_complete",
        "processing_run_id": str(result.processing_run_id),
        "requested_top_k": result.requested_top_k,
        "actual_hit_count": result.actual_hit_count,
        "chroma_hit_count": result.chroma_hit_count,
        "collection_chunk_count": result.collection_chunk_count,
        "collection": result.collection,
        "embedding_fingerprint": result.embedding_fingerprint,
        "input_representation_version": result.input_representation_version,
        "query_vector_dimension": result.query_vector_dimension,
        "retrieval": {
            "total_duration_ms": result.total_duration_ms,
            "query_embedding_duration_ms": result.query_embedding_duration_ms,
            "chroma_search_duration_ms": result.chroma_search_duration_ms,
            "postgres_resolution_duration_ms": result.postgres_resolution_duration_ms,
            "token_budget_truncated": result.token_budget_truncated,
        },
        "total_evidence_tokens": result.total_contextualized_evidence_tokens,
        **_prompt_metadata(prompt),
        "claims": [],
        "citation_validation": {"status": "pending"},
        "validation_attempts": [],
    }


def _persist_retrieval(
    result: RetrievalResult,
    question: str,
    prompt: PromptDefinition,
) -> uuid.UUID:
    """Create one query run and one deterministic hit row per selected rank."""

    query_run = QueryRun(
        document_id=result.document_id,
        pipeline_type=result.pipeline_type,
        question=question,
        top_k=result.requested_top_k,
        answer_text=None,
        answer_json=_retrieval_metadata(result, prompt),
        insufficient_evidence=False,
        retrieval_duration_ms=result.total_duration_ms,
        generation_duration_ms=None,
    )
    with SessionLocal.begin() as session:
        session.add(query_run)
        session.flush()
        session.add_all(
            RetrievalHit(
                query_run_id=query_run.id,
                chunk_id=evidence.chunk_id,
                rank=evidence.rank,
                distance=evidence.distance,
            )
            for evidence in result.evidence
        )
    return query_run.id


def _update_query_run(
    query_run_id: uuid.UUID,
    *,
    answer_text: str | None,
    answer_json: dict[str, object],
    insufficient_evidence: bool,
    generation_duration_ms: int,
) -> None:
    with SessionLocal.begin() as session:
        query_run = session.get(QueryRun, query_run_id)
        if query_run is None:
            raise RuntimeError(f"Persisted query run disappeared: {query_run_id}")
        query_run.answer_text = answer_text
        query_run.answer_json = answer_json
        query_run.insufficient_evidence = insufficient_evidence
        query_run.generation_duration_ms = generation_duration_ms


def _attempt_metadata(
    *,
    attempt_number: int,
    result: ChatAnswerResult | None,
    validation: CitationValidationResult | None,
    output_error: str | None,
    duration_ms: int,
    allowed_evidence_ids: list[str],
    validation_feedback: str | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "attempt_number": attempt_number,
        "duration_ms": duration_ms,
        "allowed_evidence_ids": allowed_evidence_ids,
        "validation_feedback": validation_feedback,
        "responses_parsing_succeeded": result is not None,
    }
    if result is not None:
        metadata.update(
            {
                "provider": result.provider,
                "deployment": result.deployment,
                "service_returned_model": result.service_model,
                "response_id": result.response_id,
                "token_usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                },
                "parsed_answer": result.answer.model_dump(),
            }
        )
    if validation is not None:
        metadata["citation_validation"] = validation.as_dict()
    if output_error is not None:
        metadata["structured_output_error"] = output_error
    return metadata


def _repair_feedback(evidence_ids: list[str]) -> str:
    allowed = ", ".join(evidence_ids)
    return (
        "The previous structured response was invalid. Use only these evidence IDs: "
        f"{allowed}. Every factual claim must cite at least one of them. "
        "Do not add page numbers. If the evidence is insufficient, return no claims."
    )


def _sum_usage(results: list[ChatAnswerResult]) -> TokenUsageResponse:
    def total(field: str) -> int | None:
        values = [getattr(result, field) for result in results]
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    return TokenUsageResponse(
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        total_tokens=total("total_tokens"),
    )


def _citation_ids(answer: AnswerResponse) -> list[str]:
    result: list[str] = []
    for claim in answer.claims:
        for citation_id in claim.citation_ids:
            if citation_id not in result:
                result.append(citation_id)
    return result


def _no_evidence_response(
    *,
    query_run_id: uuid.UUID,
    retrieval: RetrievalResult,
    question: str,
    prompt: PromptDefinition,
    total_started: float,
) -> RagResponse:
    answer_text = "The answer was not found because no evidence was retrieved."
    generation_duration_ms = 0
    metadata = _retrieval_metadata(retrieval, prompt)
    metadata.update(
        {
            "status": "completed_no_evidence",
            "claims": [],
            "citation_validation": {"status": "not_run_no_evidence"},
            "validation_attempts": [],
            "token_usage": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        }
    )
    _update_query_run(
        query_run_id,
        answer_text=answer_text,
        answer_json=metadata,
        insufficient_evidence=True,
        generation_duration_ms=generation_duration_ms,
    )
    return RagResponse(
        query_run_id=query_run_id,
        processing_run_id=retrieval.processing_run_id,
        pipeline_type=retrieval.pipeline_type,
        question=question,
        requested_top_k=retrieval.requested_top_k,
        actual_hit_count=0,
        retrieval_hits=[],
        answer=answer_text,
        claims=[],
        citation_ids=[],
        resolved_citations=[],
        insufficient_evidence=True,
        citation_validation_status="not_run_no_evidence",
        validation_attempt_count=0,
        retrieval_duration_ms=retrieval.total_duration_ms,
        generation_duration_ms=0,
        total_duration_ms=round((perf_counter() - total_started) * 1000),
        prompt_version=prompt.version,
        prompt_hash=prompt.sha256,
        schema_version=prompt.schema_version,
        provider=None,
        deployment=None,
        service_model=None,
        response_id=None,
        usage=TokenUsageResponse(),
        total_evidence_tokens=0,
    )


def answer_question(
    processing_run_id: uuid.UUID,
    question: str,
    *,
    top_k: int | None = None,
    execute: bool = True,
    settings: Settings | None = None,
) -> RagResponse:
    """Run one persisted, grounded answer flow with at most one repair retry."""

    total_started = perf_counter()
    configured_settings = settings or get_settings()
    normalized_question = question.strip()
    if not execute:
        raise ValueError("Use inspect_run_readiness for a no-call dry run")
    if not normalized_question:
        raise ValueError("Question must not be empty")
    requested_top_k = top_k if top_k is not None else configured_settings.rag_top_k
    if requested_top_k < 1 or requested_top_k > configured_settings.rag_max_top_k:
        raise ValueError(
            f"top_k must be between 1 and {configured_settings.rag_max_top_k}"
        )

    readiness = inspect_run_readiness(processing_run_id, settings=configured_settings)
    if not readiness.fully_indexed:
        raise RunNotIndexedError(
            "Processing run is not fully indexed: " + ", ".join(readiness.indexing_issues)
        )
    if not configured_settings.rag_model_ready:
        missing = ", ".join(configured_settings.azure_openai_missing_settings)
        raise ConfigurationError(f"Azure OpenAI RAG models are not configured: {missing}")
    prompt = load_rag_prompt(configured_settings.rag_prompt_version)

    retrieval = retrieve(
        processing_run_id,
        normalized_question,
        top_k=requested_top_k,
        settings=configured_settings,
    )
    query_run_id = _persist_retrieval(retrieval, normalized_question, prompt)
    if not retrieval.evidence:
        return _no_evidence_response(
            query_run_id=query_run_id,
            retrieval=retrieval,
            question=normalized_question,
            prompt=prompt,
            total_started=total_started,
        )

    try:
        provider = get_azure_openai_chat_provider(configured_settings)
    except ApplicationError as exc:
        metadata = _retrieval_metadata(retrieval, prompt)
        metadata.update(
            {
                "status": "generation_failed",
                "safe_error": {
                    "category": "chat_provider_initialization_error",
                    "error_type": type(exc).__name__,
                },
            }
        )
        _update_query_run(
            query_run_id,
            answer_text=None,
            answer_json=metadata,
            insufficient_evidence=False,
            generation_duration_ms=0,
        )
        raise
    evidence_for_model = [_model_evidence(value) for value in retrieval.evidence]
    evidence_ids = [value.evidence_id for value in retrieval.evidence]
    allowed_ids = set(evidence_ids)
    attempt_records: list[dict[str, object]] = []
    successful_results: list[ChatAnswerResult] = []
    final_result: ChatAnswerResult | None = None
    final_validation: CitationValidationResult | None = None
    generation_started = perf_counter()
    feedback: str | None = None
    total_attempts = configured_settings.chat_structured_output_retry_count + 1

    for attempt_number in range(1, total_attempts + 1):
        attempt_started = perf_counter()
        try:
            result = provider.answer(
                normalized_question,
                evidence_for_model,
                validation_feedback=feedback,
                attempt_number=attempt_number,
            )
        except ChatOutputValidationError as exc:
            attempt_records.append(
                _attempt_metadata(
                    attempt_number=attempt_number,
                    result=None,
                    validation=None,
                    output_error="invalid_structured_output",
                    duration_ms=exc.request_duration_ms,
                    allowed_evidence_ids=evidence_ids,
                    validation_feedback=feedback,
                )
            )
            feedback = _repair_feedback(evidence_ids)
            continue
        except (ChatProviderError, ConfigurationError) as exc:
            generation_duration_ms = round((perf_counter() - generation_started) * 1000)
            metadata = _retrieval_metadata(retrieval, prompt)
            metadata.update(
                {
                    "status": "generation_failed",
                    "validation_attempts": attempt_records,
                    "safe_error": {
                        "category": "chat_provider_error",
                        "error_type": type(exc).__name__,
                    },
                }
            )
            _update_query_run(
                query_run_id,
                answer_text=None,
                answer_json=metadata,
                insufficient_evidence=False,
                generation_duration_ms=generation_duration_ms,
            )
            raise

        successful_results.append(result)
        validation = validate_citations(result.answer, allowed_ids)
        attempt_records.append(
            _attempt_metadata(
                attempt_number=attempt_number,
                result=result,
                validation=validation,
                output_error=None,
                duration_ms=round((perf_counter() - attempt_started) * 1000),
                allowed_evidence_ids=evidence_ids,
                validation_feedback=feedback,
            )
        )
        if validation.valid:
            final_result = result
            final_validation = validation
            break
        feedback = _repair_feedback(evidence_ids)

    generation_duration_ms = round((perf_counter() - generation_started) * 1000)
    if final_result is None or final_validation is None:
        metadata = _retrieval_metadata(retrieval, prompt)
        metadata.update(
            {
                "status": "generation_validation_failed",
                "validation_attempts": attempt_records,
                "citation_validation": {
                    "status": "failed",
                    "note": "Citation integrity only; semantic entailment was not evaluated.",
                },
                "safe_error": {
                    "category": "generation_validation_failure",
                    "error_type": "GenerationValidationError",
                },
                "token_usage": _sum_usage(successful_results).model_dump(),
            }
        )
        _update_query_run(
            query_run_id,
            answer_text=None,
            answer_json=metadata,
            insufficient_evidence=False,
            generation_duration_ms=generation_duration_ms,
        )
        raise GenerationValidationError(
            "Structured answer or citation validation failed after the repair retry",
            query_run_id=query_run_id,
        )

    answer = final_result.answer
    citation_ids = _citation_ids(answer)
    evidence_by_id = {value.evidence_id: value for value in retrieval.evidence}
    resolved = [_resolved_citation(evidence_by_id[value]) for value in citation_ids]
    usage = _sum_usage(successful_results)
    metadata = _retrieval_metadata(retrieval, prompt)
    metadata.update(
        {
            "status": "completed",
            "model_deployment": final_result.deployment,
            "service_returned_model": final_result.service_model,
            "provider": final_result.provider,
            "response_id": final_result.response_id,
            "claims": [claim.model_dump() for claim in answer.claims],
            "citation_ids": citation_ids,
            "citation_validation": {
                "status": "valid",
                **final_validation.as_dict(),
                "note": "Citation integrity only; semantic entailment was not evaluated.",
            },
            "validation_attempts": attempt_records,
            "token_usage": usage.model_dump(),
        }
    )
    _update_query_run(
        query_run_id,
        answer_text=answer.answer,
        answer_json=metadata,
        insufficient_evidence=answer.insufficient_evidence,
        generation_duration_ms=generation_duration_ms,
    )
    return RagResponse(
        query_run_id=query_run_id,
        processing_run_id=retrieval.processing_run_id,
        pipeline_type=retrieval.pipeline_type,
        question=normalized_question,
        requested_top_k=retrieval.requested_top_k,
        actual_hit_count=retrieval.actual_hit_count,
        retrieval_hits=[_retrieval_response(value) for value in retrieval.evidence],
        answer=answer.answer,
        claims=answer.claims,
        citation_ids=citation_ids,
        resolved_citations=resolved,
        insufficient_evidence=answer.insufficient_evidence,
        citation_validation_status="valid",
        validation_attempt_count=len(attempt_records),
        retrieval_duration_ms=retrieval.total_duration_ms,
        generation_duration_ms=generation_duration_ms,
        total_duration_ms=round((perf_counter() - total_started) * 1000),
        prompt_version=final_result.prompt_version,
        prompt_hash=final_result.prompt_hash,
        schema_version=final_result.schema_version,
        provider=final_result.provider,
        deployment=final_result.deployment,
        service_model=final_result.service_model,
        response_id=final_result.response_id,
        usage=usage,
        total_evidence_tokens=retrieval.total_contextualized_evidence_tokens,
    )
