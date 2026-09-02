"""Deliberate synthetic Azure structured-chat connectivity check for future use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import ApplicationError  # noqa: E402
from app.providers.chat.azure_openai import get_azure_openai_chat_provider  # noqa: E402
from app.providers.chat.base import EvidenceForModel  # noqa: E402
from app.services.citation_validator import validate_citations  # noqa: E402


SYNTHETIC_QUESTION = "What are the synthetic readiness status and review owner?"
SYNTHETIC_EVIDENCE = [
    EvidenceForModel(
        evidence_id="E1",
        kind="text",
        evidence_class="direct source evidence",
        contextualized_text="Synthetic readiness status: green.",
    ),
    EvidenceForModel(
        evidence_id="E2",
        kind="text",
        evidence_class="direct source evidence",
        contextualized_text="Synthetic review owner: Demo Team.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Azure once with synthetic evidence using the production answer schema."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this command makes one Azure chat request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute:
        print("Chat connectivity check not run: pass --execute deliberately.", file=sys.stderr)
        return 2
    try:
        provider = get_azure_openai_chat_provider(get_settings())
        result = provider.answer(SYNTHETIC_QUESTION, SYNTHETIC_EVIDENCE)
        validation = validate_citations(result.answer, {"E1", "E2"})
        if not validation.valid:
            print(
                "Chat connectivity check failed: structured answer citation validation failed.",
                file=sys.stderr,
            )
            return 1
    except ApplicationError as exc:
        print(f"Chat connectivity check failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"Chat connectivity check failed safely: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1

    citation_ids = [
        citation_id
        for claim in result.answer.claims
        for citation_id in claim.citation_ids
    ]
    print(
        json.dumps(
            {
                "deployment_name": result.deployment,
                "service_model": result.service_model,
                "response_id": result.response_id,
                "structured_answer": result.answer.model_dump(),
                "citation_ids": citation_ids,
                "token_usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                },
                "duration_ms": result.request_duration_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
