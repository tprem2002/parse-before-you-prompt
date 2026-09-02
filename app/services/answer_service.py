"""Compatibility exports for RAG answer orchestration."""

from app.services.rag_service import GenerationValidationError, answer_question

__all__ = ["GenerationValidationError", "answer_question"]
