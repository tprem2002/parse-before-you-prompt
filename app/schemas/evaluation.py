"""Safe API contracts for controlled evaluation execution and exports."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRunRequest(BaseModel):
    """Server-controlled identifiers for one live Baseline/Docling comparison."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "document_id": "00000000-0000-4000-8000-000000000001",
                    "baseline_processing_run_id": "00000000-0000-4000-8000-000000000002",
                    "docling_processing_run_id": "00000000-0000-4000-8000-000000000003",
                    "top_k": 5,
                    "force_new": False,
                    "execute": True,
                    "confirmation": "RUN_PROJECT_AURORA_EVALUATION",
                }
            ]
        },
    )

    document_id: UUID
    baseline_processing_run_id: UUID
    docling_processing_run_id: UUID
    top_k: int = Field(default=5, ge=1, le=20)
    force_new: bool = False
    execute: bool = True
    confirmation: str = Field(min_length=1, max_length=80)


class EvaluationRunResponse(BaseModel):
    """Progress, measured summaries, and optional per-question results."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    evaluation_id: UUID
    document_id: UUID
    status: str
    reused: bool
    progress_percent: float
    current_question_id: str | None
    completed_case_count: int
    total_case_count: int
    question_count: int
    answerable_count: int
    unsupported_count: int
    started_at: str | None
    completed_at: str | None
    elapsed_duration_ms: int | None
    duration_ms: int | None
    configuration: dict[str, Any]
    evaluation_version: str | None
    metric_definition_version: str | None
    ground_truth_sha256: str | None
    source_sha256: str | None
    pipeline_summaries: dict[str, Any]
    category_summaries: list[dict[str, Any]]
    failure_summary: dict[str, Any]
    processing_metrics: dict[str, Any]
    limitations: list[str]
    export_urls: dict[str, str]
    safe_error: dict[str, Any] | None
    results: list[dict[str, Any]] | None
    created_at: str
