"""Manual dry-run and exactly-once controlled evaluation command."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.evaluation_service import (
    TERMINAL_REUSABLE_STATUSES,
    build_evaluation_plan,
    create_or_reuse_evaluation,
    evaluation_view,
    execute_evaluation,
)


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid UUID: {value}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute the fixed Project Aurora Baseline/Docling evaluation. "
            "No prompt, model, endpoint, or ground-truth override is accepted."
        )
    )
    parser.add_argument("--document-id", type=_uuid, required=True)
    parser.add_argument("--baseline-processing-run-id", type=_uuid, required=True)
    parser.add_argument("--docling-processing-run-id", type=_uuid, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-new", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize all live Azure query-embedding and chat calls.",
    )
    return parser


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main() -> int:
    args = _parser().parse_args()
    settings = get_settings()
    try:
        plan = build_evaluation_plan(
            document_id=args.document_id,
            baseline_processing_run_id=args.baseline_processing_run_id,
            docling_processing_run_id=args.docling_processing_run_id,
            top_k=args.top_k,
            settings=settings,
        )
    except Exception as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        payload = plan.as_dict()
        payload["mode"] = "dry_run"
        payload["notice"] = "No Chroma query or Azure model call occurred. No EvaluationRun was created."
        _print_json(payload)
        return 0

    print(
        f"Executing {len(plan.ground_truth.questions)} questions x 2 pipelines "
        f"({plan.total_cases} cases) sequentially."
    )
    try:
        evaluation_id, reused = create_or_reuse_evaluation(plan, force_new=args.force_new)
        if not reused:
            def progress(item: dict[str, Any]) -> None:
                print(
                    f"[{item['completed_case_count']}/{item['total_case_count']}] "
                    f"{item['question_id']} {item['pipeline']} {item['result_status']}"
                )

            execute_evaluation(
                evaluation_id,
                plan=plan,
                settings=settings,
                progress_callback=progress,
            )
        view = evaluation_view(evaluation_id, include_results=False, reused=reused)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 3

    print(f"Evaluation ID: {evaluation_id}")
    print(f"Reused: {str(reused).lower()}")
    print(f"Status: {view['status']}")
    _print_json({"pipeline_summaries": view["pipeline_summaries"]})
    if view["status"] in TERMINAL_REUSABLE_STATUSES:
        relative_root = Path("data") / "artifacts" / "evaluations" / str(evaluation_id)
        print(f"CSV: {(relative_root / f'evaluation-{evaluation_id}.csv').as_posix()}")
        print(f"JSON: {(relative_root / f'evaluation-{evaluation_id}.json').as_posix()}")
    else:
        print("Exports: unavailable because the evaluation did not complete successfully.")
    return 0 if view["status"] in TERMINAL_REUSABLE_STATUSES else 4


if __name__ == "__main__":
    raise SystemExit(main())
