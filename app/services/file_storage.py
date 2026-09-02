"""Safe local storage for immutable uploads and generated text artifacts."""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.core.config import Settings, get_settings
from app.services.hashing import sha256_file


_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str | None) -> str:
    """Return a conservative basename safe for local filesystem storage."""

    candidate = Path(filename or "upload.pdf").name
    candidate = unicodedata.normalize("NFKC", candidate)
    candidate = _SAFE_FILENAME_PATTERN.sub("_", candidate).strip(" ._")
    if not candidate:
        candidate = "upload.pdf"

    stem = Path(candidate).stem[:160].rstrip(" ._") or "upload"
    suffix = Path(candidate).suffix.lower()
    return f"{stem}{suffix}"


class FileStorage:
    """Store files beneath the configured upload and artifact roots."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.upload_root = self.settings.upload_root.resolve()
        self.artifact_root = self.settings.artifact_root.resolve()

    def store_upload(self, content: bytes, sha256: str, filename: str) -> Path:
        """Persist exact uploaded bytes under a content-addressed directory."""

        target = self._within_root(self.upload_root, self.upload_root / sha256 / filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != sha256:
                raise RuntimeError(f"Existing upload has an unexpected digest: {target}")
            return target

        self._atomic_write_bytes(target, content)
        if sha256_file(target) != sha256:
            target.unlink(missing_ok=True)
            raise RuntimeError("Stored upload digest does not match the uploaded bytes")
        return target

    def write_baseline_text(
        self,
        *,
        document_id: uuid.UUID,
        processing_run_id: uuid.UUID,
        text: str,
    ) -> Path:
        """Persist a baseline plain-text export as UTF-8."""

        target = self._within_root(
            self.artifact_root,
            self.artifact_root / str(document_id) / str(processing_run_id) / "baseline.txt",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_bytes(target, text.encode("utf-8"))
        return target

    def artifact_directory(
        self, *, document_id: uuid.UUID, processing_run_id: uuid.UUID
    ) -> Path:
        """Return and create the private output directory for one processing run."""

        target = self._within_root(
            self.artifact_root,
            self.artifact_root / str(document_id) / str(processing_run_id),
        )
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_artifact_bytes(
        self,
        *,
        document_id: uuid.UUID,
        processing_run_id: uuid.UUID,
        relative_path: str | Path,
        content: bytes,
    ) -> Path:
        """Atomically write one generated artifact beneath a run directory."""

        run_directory = self.artifact_directory(
            document_id=document_id, processing_run_id=processing_run_id
        )
        target = self._within_root(run_directory, run_directory / relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_bytes(target, content)
        return target

    def write_artifact_text(
        self,
        *,
        document_id: uuid.UUID,
        processing_run_id: uuid.UUID,
        relative_path: str | Path,
        text: str,
    ) -> Path:
        """Atomically write a UTF-8 artifact beneath a run directory."""

        return self.write_artifact_bytes(
            document_id=document_id,
            processing_run_id=processing_run_id,
            relative_path=relative_path,
            content=text.encode("utf-8"),
        )

    def write_png_artifact(
        self,
        *,
        document_id: uuid.UUID,
        processing_run_id: uuid.UUID,
        relative_path: str | Path,
        image: Image.Image,
    ) -> Path:
        """Encode a Pillow image as PNG and store it atomically."""

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return self.write_artifact_bytes(
            document_id=document_id,
            processing_run_id=processing_run_id,
            relative_path=relative_path,
            content=buffer.getvalue(),
        )

    @staticmethod
    def _within_root(root: Path, target: Path) -> Path:
        resolved = target.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Storage target escapes configured root: {target}")
        return resolved

    @staticmethod
    def _atomic_write_bytes(target: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(content)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
