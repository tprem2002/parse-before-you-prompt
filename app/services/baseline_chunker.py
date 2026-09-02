"""Fixed-token baseline chunking with deterministic page-range tracking."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from tiktoken import Encoding

from app.core.config import Settings, get_settings
from app.services.baseline_parser import BaselineParseResult


@dataclass(frozen=True, slots=True)
class BaselineChunk:
    """A fixed token window and the pages represented by its tokens."""

    ordinal: int
    text: str
    token_count: int
    page_start: int
    page_end: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TokenizerResolution:
    """The selected tokenizer plus the configuration that selected it."""

    encoding: Encoding
    model_hint: str
    fallback_encoding: str
    used_fallback: bool


def resolve_tokenizer(settings: Settings | None = None) -> TokenizerResolution:
    """Resolve the model tokenizer and explicitly fall back when unknown."""

    settings = settings or get_settings()
    try:
        encoding = tiktoken.encoding_for_model(settings.tiktoken_model_hint)
        used_fallback = False
    except KeyError:
        encoding = tiktoken.get_encoding(settings.tiktoken_fallback_encoding)
        used_fallback = True
    return TokenizerResolution(
        encoding=encoding,
        model_hint=settings.tiktoken_model_hint,
        fallback_encoding=settings.tiktoken_fallback_encoding,
        used_fallback=used_fallback,
    )


def chunk_baseline(
    parsed: BaselineParseResult,
    settings: Settings | None = None,
) -> tuple[BaselineChunk, ...]:
    """Create overlapping fixed-token chunks while retaining page ranges."""

    settings = settings or get_settings()
    maximum = settings.chunk_max_tokens
    overlap = settings.baseline_chunk_overlap_tokens
    if overlap >= maximum:
        raise ValueError("Baseline chunk overlap must be smaller than the maximum chunk size")

    resolution = resolve_tokenizer(settings)
    all_tokens: list[int] = []
    page_ranges: list[tuple[int, int, int]] = []
    for page in parsed.pages:
        start = len(all_tokens)
        page_tokens = resolution.encoding.encode(page.artifact_segment, disallowed_special=())
        all_tokens.extend(page_tokens)
        page_ranges.append((page.page_no, start, len(all_tokens)))

    if not all_tokens:
        return ()

    chunks: list[BaselineChunk] = []
    step = maximum - overlap
    for ordinal, start in enumerate(range(0, len(all_tokens), step)):
        end = min(start + maximum, len(all_tokens))
        window = tuple(all_tokens[start:end])
        represented_pages = [
            page_no
            for page_no, page_token_start, page_token_end in page_ranges
            if start < page_token_end and end > page_token_start
        ]
        if not represented_pages:
            raise RuntimeError("Chunk token window could not be mapped to a source page")
        chunks.append(
            BaselineChunk(
                ordinal=ordinal,
                text=resolution.encoding.decode(window),
                token_count=len(window),
                page_start=min(represented_pages),
                page_end=max(represented_pages),
                token_ids=window,
            )
        )
        if end == len(all_tokens):
            break
    return tuple(chunks)
