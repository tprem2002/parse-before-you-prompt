"""Realistic page-by-page plain-text extraction using PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


PAGE_SEPARATOR_TEMPLATE = "===== PAGE {page_no} =====\n"


class InvalidPdfError(ValueError):
    """Raised when uploaded bytes cannot be treated as an unencrypted PDF."""


@dataclass(frozen=True, slots=True)
class BaselinePage:
    """Plain text and artifact segment for one source page."""

    page_no: int
    text: str
    artifact_segment: str


@dataclass(frozen=True, slots=True)
class BaselineParseResult:
    """Complete baseline extraction with explicit page boundaries."""

    pages: tuple[BaselinePage, ...]
    full_text: str

    @property
    def page_count(self) -> int:
        return len(self.pages)


def inspect_pdf_bytes(content: bytes) -> int:
    """Validate uploaded PDF bytes and return their page count."""

    if not content or not content.lstrip().startswith(b"%PDF-"):
        raise InvalidPdfError("Uploaded content does not have a PDF signature")
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise InvalidPdfError("Password-protected PDFs are not supported")
            if not document.is_pdf or document.page_count < 1:
                raise InvalidPdfError("Uploaded PDF has no readable pages")
            return document.page_count
    except InvalidPdfError:
        raise
    except Exception as exc:
        raise InvalidPdfError("Uploaded content is not a readable PDF") from exc


def parse_pdf(path: Path) -> BaselineParseResult:
    """Extract normal PyMuPDF text without OCR or structural reconstruction."""

    pages: list[BaselinePage] = []
    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise InvalidPdfError("Password-protected PDFs are not supported")
            for page_index, page in enumerate(document):
                page_no = page_index + 1
                plain_text = page.get_text("text", sort=False)
                if plain_text and not plain_text.endswith("\n"):
                    plain_text += "\n"
                prefix = "" if page_index == 0 else "\n"
                artifact_segment = (
                    prefix + PAGE_SEPARATOR_TEMPLATE.format(page_no=page_no) + plain_text
                )
                pages.append(
                    BaselinePage(
                        page_no=page_no,
                        text=plain_text,
                        artifact_segment=artifact_segment,
                    )
                )
    except InvalidPdfError:
        raise
    except Exception as exc:
        raise InvalidPdfError(f"PyMuPDF could not parse {path.name}") from exc

    if not pages:
        raise InvalidPdfError("Uploaded PDF has no readable pages")
    return BaselineParseResult(pages=tuple(pages), full_text="".join(p.artifact_segment for p in pages))
