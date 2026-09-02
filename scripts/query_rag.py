"""Dry-run or explicitly execute one RAG question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import ApplicationError  # noqa: E402
from app.services.rag_service import GenerationValidationError, answer_question  # noqa: E402
from app.services.retrieval_service import inspect_run_readiness  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or execute one grounded RAG question.")
    parser.add_argument("--processing-run-id", required=True, type=UUID)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    top_k = args.top_k if args.top_k is not None else settings.rag_top_k
    if top_k < 1 or top_k > settings.rag_max_top_k:
        print(f"RAG request failed: top_k must be between 1 and {settings.rag_max_top_k}", file=sys.stderr)
        return 2
    if not args.question.strip():
        print("RAG request failed: question must not be empty", file=sys.stderr)
        return 2

    try:
        if args.dry_run:
            readiness = inspect_run_readiness(args.processing_run_id, settings=settings)
            print(json.dumps(readiness.as_dict(settings, requested_top_k=top_k), indent=2, sort_keys=True))
            return 0
        result = answer_question(
            args.processing_run_id,
            args.question,
            top_k=top_k,
            execute=True,
            settings=settings,
        )
    except GenerationValidationError as exc:
        print(
            f"RAG request failed: generation_validation_failure; query_run_id={exc.query_run_id}",
            file=sys.stderr,
        )
        return 1
    except (ApplicationError, LookupError, ValueError) as exc:
        print(f"RAG request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"RAG request failed safely with unexpected error type: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
