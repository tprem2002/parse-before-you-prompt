"""Dry-run-by-default wrapper for the narrow project-owned reset service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.reset_service import reset_demo


CONFIRMATION = "RESET_PROJECT_AURORA_DEMO"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute cleanup of registered project data. Model cache is never removed."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    args = parser.parse_args()
    if args.execute and args.confirmation != CONFIRMATION:
        parser.error(f"--execute requires --confirmation {CONFIRMATION}")
    settings = get_settings()
    if args.execute and not settings.allow_demo_reset:
        print("Reset execution is disabled. Set ALLOW_DEMO_RESET=true deliberately, then retry.", file=sys.stderr)
        return 2
    try:
        response = reset_demo(dry_run=not args.execute, settings=settings)
    except Exception as exc:
        print(f"Reset failed safely: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(response.model_dump(mode="json"), indent=2))
    if not args.execute:
        print("Dry run only. No database row, Chroma collection, artifact, upload, or model cache was removed.")
    return 0 if not response.failures else 4


if __name__ == "__main__":
    raise SystemExit(main())
