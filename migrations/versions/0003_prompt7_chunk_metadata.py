"""Add chunking and provenance metadata.

Revision ID: 0003_prompt7_chunk_metadata
Revises: 0002_conversion_manifest
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003_prompt7_chunk_metadata"
down_revision: str | None = "0002_conversion_manifest"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_OLD_CHUNK_KINDS = (
    "'text', 'heading', 'list', 'table', 'picture', 'formula', 'code', "
    "'mixed', 'unknown'"
)
_NEW_CHUNK_KINDS = (
    "'text', 'heading', 'list', 'table', 'picture', 'formula', 'code', "
    "'caption', 'footnote', 'form', 'key_value', 'mixed', 'unknown'"
)


def upgrade() -> None:
    """Extend chunks and provenance while preserving existing baseline rows."""

    op.add_column("chunks", sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("chunks", sa.Column("raw_token_count", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("contextualized_token_count", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("max_token_count", sa.Integer(), nullable=True))
    op.add_column(
        "chunks",
        sa.Column(
            "content_classification",
            sa.String(length=16),
            server_default=sa.text("'source'"),
            nullable=False,
        ),
    )
    op.add_column("chunks", sa.Column("chunking_fingerprint", sa.String(length=64), nullable=True))
    op.add_column(
        "chunks",
        sa.Column(
            "serializer_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "chunk_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column("chunks", sa.Column("header_repetition_status", sa.String(length=16), nullable=True))
    op.add_column(
        "chunks",
        sa.Column("overflow", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )

    op.execute(
        "UPDATE chunks AS c SET document_id = r.document_id "
        "FROM processing_runs AS r WHERE r.id = c.processing_run_id"
    )
    op.execute("UPDATE chunks SET raw_token_count = token_count")
    op.execute("UPDATE chunks SET contextualized_token_count = token_count")
    op.alter_column("chunks", "document_id", nullable=False)
    op.alter_column("chunks", "raw_token_count", nullable=False)
    op.alter_column("chunks", "contextualized_token_count", nullable=False)
    op.create_foreign_key(
        op.f("fk_chunks_document_id_documents"),
        "chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(op.f("ix_chunks_document_id"), "chunks", ["document_id"])
    op.create_index(op.f("ix_chunks_chunking_fingerprint"), "chunks", ["chunking_fingerprint"])
    op.create_check_constraint(op.f("ck_chunks_raw_token_count"), "chunks", "raw_token_count >= 0")
    op.create_check_constraint(
        op.f("ck_chunks_contextualized_token_count"),
        "chunks",
        "contextualized_token_count >= 0",
    )
    op.create_check_constraint(
        op.f("ck_chunks_max_token_count"),
        "chunks",
        "max_token_count IS NULL OR max_token_count >= 1",
    )
    op.create_check_constraint(
        op.f("ck_chunks_content_classification"),
        "chunks",
        "content_classification IN ('source', 'derived', 'mixed')",
    )
    op.create_check_constraint(
        op.f("ck_chunks_header_repetition_status"),
        "chunks",
        "header_repetition_status IS NULL OR header_repetition_status IN "
        "('repeated', 'not_repeated', 'not_applicable', 'unknown')",
    )
    op.drop_constraint(op.f("ck_chunks_kind"), "chunks", type_="check")
    op.create_check_constraint(
        op.f("ck_chunks_kind"), "chunks", f"kind IN ({_NEW_CHUNK_KINDS})"
    )

    op.add_column(
        "provenance_records",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "provenance_records",
        sa.Column("processing_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "provenance_records",
        sa.Column(
            "evidence_role",
            sa.String(length=32),
            server_default=sa.text("'direct_source_text'"),
            nullable=False,
        ),
    )
    op.execute(
        "UPDATE provenance_records AS p SET document_id = c.document_id, "
        "processing_run_id = c.processing_run_id FROM chunks AS c WHERE c.id = p.chunk_id"
    )
    op.alter_column("provenance_records", "document_id", nullable=False)
    op.alter_column("provenance_records", "processing_run_id", nullable=False)
    op.create_foreign_key(
        op.f("fk_provenance_records_document_id_documents"),
        "provenance_records",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        op.f("fk_provenance_records_processing_run_id_processing_runs"),
        "provenance_records",
        "processing_runs",
        ["processing_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(op.f("ix_provenance_records_document_id"), "provenance_records", ["document_id"])
    op.create_index(
        op.f("ix_provenance_records_processing_run_id"),
        "provenance_records",
        ["processing_run_id"],
    )
    op.create_check_constraint(
        op.f("ck_provenance_records_evidence_role"),
        "provenance_records",
        "evidence_role IN ('direct_source_text', 'source_image_region', "
        "'derived_visual_anchor')",
    )


def downgrade() -> None:
    """Remove chunking/provenance fields after restoring the prior kind constraint."""

    op.drop_constraint(op.f("ck_provenance_records_evidence_role"), "provenance_records", type_="check")
    op.drop_index(op.f("ix_provenance_records_processing_run_id"), table_name="provenance_records")
    op.drop_index(op.f("ix_provenance_records_document_id"), table_name="provenance_records")
    op.drop_constraint(
        op.f("fk_provenance_records_processing_run_id_processing_runs"),
        "provenance_records",
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f("fk_provenance_records_document_id_documents"),
        "provenance_records",
        type_="foreignkey",
    )
    op.drop_column("provenance_records", "evidence_role")
    op.drop_column("provenance_records", "processing_run_id")
    op.drop_column("provenance_records", "document_id")

    op.drop_constraint(op.f("ck_chunks_kind"), "chunks", type_="check")
    op.create_check_constraint(op.f("ck_chunks_kind"), "chunks", f"kind IN ({_OLD_CHUNK_KINDS})")
    op.drop_constraint(op.f("ck_chunks_header_repetition_status"), "chunks", type_="check")
    op.drop_constraint(op.f("ck_chunks_content_classification"), "chunks", type_="check")
    op.drop_constraint(op.f("ck_chunks_max_token_count"), "chunks", type_="check")
    op.drop_constraint(op.f("ck_chunks_contextualized_token_count"), "chunks", type_="check")
    op.drop_constraint(op.f("ck_chunks_raw_token_count"), "chunks", type_="check")
    op.drop_index(op.f("ix_chunks_chunking_fingerprint"), table_name="chunks")
    op.drop_index(op.f("ix_chunks_document_id"), table_name="chunks")
    op.drop_constraint(op.f("fk_chunks_document_id_documents"), "chunks", type_="foreignkey")
    op.drop_column("chunks", "overflow")
    op.drop_column("chunks", "header_repetition_status")
    op.drop_column("chunks", "chunk_metadata")
    op.drop_column("chunks", "serializer_metadata")
    op.drop_column("chunks", "chunking_fingerprint")
    op.drop_column("chunks", "content_classification")
    op.drop_column("chunks", "max_token_count")
    op.drop_column("chunks", "contextualized_token_count")
    op.drop_column("chunks", "raw_token_count")
    op.drop_column("chunks", "document_id")
