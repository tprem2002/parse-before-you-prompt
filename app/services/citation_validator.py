"""Deterministic citation-integrity checks over supplied evidence IDs."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.rag import AnswerResponse


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    valid: bool
    error_codes: tuple[str, ...]
    messages: tuple[str, ...]
    invalid_citation_ids: tuple[str, ...]
    uncited_claim_indexes: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "error_codes": list(self.error_codes),
            "messages": list(self.messages),
            "invalid_citation_ids": list(self.invalid_citation_ids),
            "uncited_claim_indexes": list(self.uncited_claim_indexes),
        }


def validate_citations(
    answer: AnswerResponse,
    supplied_evidence_ids: set[str],
) -> CitationValidationResult:
    """Validate evidence-ID integrity; this does not prove semantic entailment."""

    codes: list[str] = []
    messages: list[str] = []
    invalid_ids: list[str] = []
    uncited_indexes: list[int] = []

    if not answer.answer.strip():
        codes.append("empty_answer")
        messages.append("The structured answer text is empty.")

    if answer.insufficient_evidence:
        if answer.claims:
            codes.append("abstention_claims_present")
            messages.append("An insufficient-evidence response must not contain factual claims.")
    elif not answer.claims:
        codes.append("claims_required")
        messages.append("A supported answer must contain at least one cited claim.")

    for index, claim in enumerate(answer.claims):
        if not claim.text.strip():
            codes.append("empty_claim_text")
            messages.append(f"Claim {index} has empty text.")
        if not claim.citation_ids:
            codes.append("uncited_claim")
            uncited_indexes.append(index)
            messages.append(f"Claim {index} has no citation IDs.")
        if len(set(claim.citation_ids)) != len(claim.citation_ids):
            codes.append("duplicate_citation_id")
            messages.append(f"Claim {index} repeats a citation ID.")
        for citation_id in claim.citation_ids:
            if citation_id not in supplied_evidence_ids and citation_id not in invalid_ids:
                invalid_ids.append(citation_id)

    if invalid_ids:
        codes.append("unknown_citation_id")
        messages.append("One or more citation IDs were not supplied with the evidence.")

    return CitationValidationResult(
        valid=not codes,
        error_codes=tuple(dict.fromkeys(codes)),
        messages=tuple(messages),
        invalid_citation_ids=tuple(invalid_ids),
        uncited_claim_indexes=tuple(uncited_indexes),
    )
