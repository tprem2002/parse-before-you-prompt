"""Deliberate synthetic Azure embedding connectivity check for future use."""

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
from app.providers.embeddings.azure_openai import (  # noqa: E402
    get_azure_openai_embedding_provider,
)


SYNTHETIC_TEXT = "Parse Before You Prompt embedding connectivity check."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Azure once with synthetic text; no document content is used."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this command makes one Azure embedding request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute:
        print("Connectivity check not run: pass --execute deliberately.", file=sys.stderr)
        return 2
    try:
        provider = get_azure_openai_embedding_provider(get_settings())
        result = provider.embed_documents([SYNTHETIC_TEXT], input_ids=["synthetic-connectivity"])
    except ApplicationError as exc:
        print(f"Embedding connectivity check failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"Embedding connectivity check failed safely: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "deployment_name": result.deployment_name,
                "service_model": result.service_model,
                "vector_dimension": result.vector_dimension,
                "usage_prompt_tokens": result.usage_prompt_tokens,
                "usage_total_tokens": result.usage_total_tokens,
                "request_ids": list(result.request_ids),
                "request_duration_ms": result.request_duration_ms,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
