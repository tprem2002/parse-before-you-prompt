"""Plan or explicitly execute indexing for one completed run."""

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
from app.services.embedding_index_service import index_processing_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute Azure external-vector indexing for a completed run."
    )
    parser.add_argument("--processing-run-id", required=True, type=UUID)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Explicitly upsert the current fingerprint even when vector metadata exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    try:
        result = index_processing_run(
            args.processing_run_id,
            execute=args.execute,
            force_reindex=args.force_reindex,
            settings=settings,
        )
    except (ApplicationError, LookupError, ValueError) as exc:
        print(f"Indexing request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"Indexing request failed safely with unexpected error type: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
