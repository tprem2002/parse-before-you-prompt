"""PDF upload, deduplication, and document lookup endpoints."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document
from app.db.repositories import DocumentRepository
from app.db.session import get_db
from app.schemas.documents import DocumentResponse, UploadResponse
from app.schemas.common import error_responses
from app.services.api_view_service import document_view
from app.services.baseline_parser import InvalidPdfError, inspect_pdf_bytes
from app.services.file_storage import FileStorage, sanitize_filename
from app.services.hashing import sha256_bytes


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    responses=error_responses(404, 413, 415, 422, 500),
)
document_repository = DocumentRepository()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_200_OK: {
            "model": UploadResponse,
            "description": "Identical PDF reused.",
        }
    },
    summary="Upload and validate a PDF",
)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
) -> UploadResponse:
    """Store a valid PDF once, keyed by the SHA-256 of its exact bytes."""

    settings = get_settings()
    safe_filename = sanitize_filename(file.filename)
    if Path(safe_filename).suffix.lower() != ".pdf" or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only files named as PDF with content type application/pdf are supported",
        )

    maximum_bytes = settings.docling_max_file_size_mb * 1024 * 1024
    content = bytearray()
    try:
        while len(content) <= maximum_bytes:
            block = await file.read(min(1024 * 1024, maximum_bytes + 1 - len(content)))
            if not block:
                break
            content.extend(block)
    finally:
        await file.close()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded PDF is empty")
    if len(content) > maximum_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"PDF exceeds the configured {settings.docling_max_file_size_mb} MB limit",
        )

    original_bytes = bytes(content)
    digest = sha256_bytes(original_bytes)
    existing = document_repository.get_by_sha256(session, digest)
    if existing is not None:
        response.status_code = status.HTTP_200_OK
        return UploadResponse(
            document=document_view(existing, session),
            duplicate=True,
            reused=True,
            message=f"Identical PDF already exists as document {existing.id}",
        )

    try:
        page_count = inspect_pdf_bytes(original_bytes)
    except InvalidPdfError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if page_count > settings.docling_max_pages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF exceeds the configured {settings.docling_max_pages}-page limit",
        )

    storage_path = FileStorage(settings).store_upload(original_bytes, digest, safe_filename)
    document = Document(
        filename=safe_filename,
        display_name=Path(safe_filename).stem,
        mime_type="application/pdf",
        sha256=digest,
        file_size_bytes=len(original_bytes),
        storage_path=str(storage_path),
        page_count=page_count,
    )
    try:
        document_repository.add(session, document)
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = document_repository.get_by_sha256(session, digest)
        if existing is None:
            raise
        response.status_code = status.HTTP_200_OK
        return UploadResponse(
            document=document_view(existing, session),
            duplicate=True,
            reused=True,
            message=f"Identical PDF already exists as document {existing.id}",
        )
    session.refresh(document)
    response.status_code = status.HTTP_201_CREATED
    return UploadResponse(
        document=document_view(document, session),
        duplicate=False,
        reused=False,
        message="PDF uploaded successfully",
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> list[DocumentResponse]:
    """List uploaded documents newest first."""

    return [
        document_view(document, session)
        for document in document_repository.list(session, limit=limit, offset=offset)
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: UUID, session: Session = Depends(get_db)) -> DocumentResponse:
    """Return one uploaded document."""

    document = document_repository.get(session, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document_view(document, session)
