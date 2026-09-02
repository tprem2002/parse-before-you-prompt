"""Narrow reset planning and execution for project-owned demo resources."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from app.core.config import Settings, get_settings
from app.db.models import (
    Artifact,
    Chunk,
    Document,
    EvaluationRun,
    ProcessingRun,
    ProvenanceRecord,
    QueryRun,
    RetrievalHit,
)
from app.db.session import SessionLocal
from app.schemas.admin import ResetDemoResponse
from app.services.chroma_service import get_chroma_service


def _safe_registered_path(path: str, roots: tuple[Path, ...]) -> tuple[Path, str]:
    resolved = Path(path).resolve()
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved, str(resolved.relative_to(root)).replace("\\", "/")
    raise ValueError("registered file is outside project-managed storage")


def reset_demo(
    *,
    dry_run: bool,
    settings: Settings | None = None,
) -> ResetDemoResponse:
    configured = settings or get_settings()
    roots = (configured.upload_root.resolve(), configured.artifact_root.resolve())
    forbidden = {Path(root.anchor) for root in roots} | {Path.cwd().resolve(), Path.home().resolve()}
    if any(root in forbidden for root in roots):
        raise ValueError("configured reset roots are too broad")

    with SessionLocal() as session:
        models = (
            ("documents", Document),
            ("processing_runs", ProcessingRun),
            ("chunks", Chunk),
            ("provenance_records", ProvenanceRecord),
            ("artifacts", Artifact),
            ("query_runs", QueryRun),
            ("retrieval_hits", RetrievalHit),
            ("evaluation_runs", EvaluationRun),
        )
        resources = {
            name: int(session.scalar(select(func.count()).select_from(model)) or 0)
            for name, model in models
        }
        registered = [item.storage_path for item in session.scalars(select(Artifact))]
        registered.extend(item.storage_path for item in session.scalars(select(Document)))
        document_ids = list(session.scalars(select(Document.id)))
        active_evaluations = [
            item.id
            for item in session.scalars(select(EvaluationRun))
            if (item.summary_json or {}).get("status") in {"queued", "running"}
        ]
    if active_evaluations:
        raise ValueError("demo reset is unavailable while an evaluation is queued or running")

    evaluation_output_root = (configured.artifact_root / "evaluations").resolve()
    if evaluation_output_root == configured.artifact_root.resolve() or configured.artifact_root.resolve() not in evaluation_output_root.parents:
        raise ValueError("evaluation output reset scope is too broad")
    if evaluation_output_root.exists():
        registered.extend(str(item) for item in evaluation_output_root.rglob("*") if item.is_file())

    safe_files: dict[Path, str] = {}
    failures: list[str] = []
    for raw in registered:
        try:
            resolved, relative = _safe_registered_path(raw, roots)
            safe_files[resolved] = relative
        except ValueError:
            failures.append("One registered file was outside project-managed storage.")

    chroma = get_chroma_service(configured)
    collection_names = sorted(
        item.name if hasattr(item, "name") else str(item)
        for item in chroma._get_client().list_collections()
        if (item.name if hasattr(item, "name") else str(item)).startswith("pbtp_")
    )
    if dry_run:
        return ResetDemoResponse(
            dry_run=True,
            executed=False,
            reset_enabled=configured.allow_demo_reset,
            resources=resources,
            chroma_collections=collection_names,
            managed_files=sorted(safe_files.values()),
            failures=failures,
        )

    for name in collection_names:
        try:
            chroma._get_client().delete_collection(name=name)
        except Exception:
            failures.append(f"Chroma collection cleanup failed for {name}.")
    with SessionLocal.begin() as session:
        for document_id in document_ids:
            document = session.get(Document, document_id)
            if document is not None:
                session.delete(document)
    for path, relative in safe_files.items():
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except OSError:
            failures.append(f"Managed file cleanup failed for {relative}.")
    for root in roots:
        if not root.exists():
            continue
        for directory in sorted(
            (item for item in root.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
    return ResetDemoResponse(
        dry_run=False,
        executed=True,
        reset_enabled=True,
        resources=resources,
        chroma_collections=collection_names,
        managed_files=sorted(safe_files.values()),
        failures=failures,
    )
