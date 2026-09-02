"""Versioned prompt loading with stable audit hashes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import ConfigurationError


PROMPT_FILES = {
    "parse-before-you-prompt-rag-v1": "parse-before-you-prompt-rag-v1.txt",
}
ANSWER_SCHEMA_VERSION = "parse-before-you-prompt-answer-v1"


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    version: str
    text: str
    sha256: str
    schema_version: str


def load_rag_prompt(version: str) -> PromptDefinition:
    """Load one explicitly supported prompt without depending on the working directory."""

    filename = PROMPT_FILES.get(version)
    if filename is None:
        raise ConfigurationError(f"Unsupported RAG_PROMPT_VERSION: {version}")
    text = (Path(__file__).resolve().parent / filename).read_text(encoding="utf-8").strip()
    return PromptDefinition(
        version=version,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        schema_version=ANSWER_SCHEMA_VERSION,
    )
