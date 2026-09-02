"""Create the initial document processing and RAG audit schema.

Revision ID: 0001_initial_schema
Revises: None
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all initial tables, constraints, and lookup indexes."""

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=127), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("sha256", name="uq_documents_sha256"),
    )
    op.create_table(
        "processing_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pipeline_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=False),
        sa.Column(
            "configuration_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "pipeline_type IN ('baseline', 'docling_standard', 'docling_granite_vlm')",
            name=op.f("ck_processing_runs_pipeline_type"),
        ),
        sa.CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name=op.f("ck_processing_runs_progress_percent"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'partial', 'failed')",
            name=op.f("ck_processing_runs_status"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name="fk_processing_runs_document_id_documents", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_processing_runs"),
    )
    op.create_index("ix_processing_runs_document_id", "processing_runs", ["document_id"])
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("processing_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("chunk_role", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column(
            "section_path",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "captions",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "doc_item_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("table_ref", sa.Text(), nullable=True),
        sa.Column("picture_ref", sa.Text(), nullable=True),
        sa.Column("is_derived_content", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("vector_collection", sa.String(length=255), nullable=True),
        sa.Column("vector_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "chunk_role IN ('hierarchical_inspection', 'vector_index')",
            name=op.f("ck_chunks_chunk_role"),
        ),
        sa.CheckConstraint(
            "kind IN ('text', 'heading', 'list', 'table', 'picture', 'formula', 'code', 'mixed', 'unknown')",
            name=op.f("ck_chunks_kind"),
        ),
        sa.CheckConstraint("ordinal >= 0", name=op.f("ck_chunks_ordinal")),
        sa.CheckConstraint("token_count >= 0", name=op.f("ck_chunks_token_count")),
        sa.ForeignKeyConstraint(
            ["processing_run_id"],
            ["processing_runs.id"],
            name="fk_chunks_processing_run_id_processing_runs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint(
            "processing_run_id", "chunk_role", "ordinal", name="run_role_ordinal"
        ),
    )
    op.create_index("ix_chunks_processing_run_id", "chunks", ["processing_run_id"])
    op.create_table(
        "provenance_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("doc_item_ref", sa.Text(), nullable=False),
        sa.Column("page_no", sa.Integer(), nullable=False),
        sa.Column("bbox_left", sa.Float(), nullable=True),
        sa.Column("bbox_top", sa.Float(), nullable=True),
        sa.Column("bbox_right", sa.Float(), nullable=True),
        sa.Column("bbox_bottom", sa.Float(), nullable=True),
        sa.Column("coordinate_origin", sa.String(length=32), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "char_end IS NULL OR char_end >= 0",
            name=op.f("ck_provenance_records_char_end"),
        ),
        sa.CheckConstraint(
            "char_start IS NULL OR char_start >= 0",
            name=op.f("ck_provenance_records_char_start"),
        ),
        sa.CheckConstraint("page_no >= 1", name=op.f("ck_provenance_records_page_no")),
        sa.ForeignKeyConstraint(
            ["chunk_id"], ["chunks.id"], name="fk_provenance_records_chunk_id_chunks", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_provenance_records"),
    )
    op.create_index("ix_provenance_records_chunk_id", "provenance_records", ["chunk_id"])
    op.create_table(
        "artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("processing_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_type", sa.String(length=32), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("page_no", sa.Integer(), nullable=True),
        sa.Column("doc_item_ref", sa.Text(), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "artifact_type IN ('original_pdf', 'docling_json', 'markdown', 'baseline_text', 'page_image', 'picture_image', 'table_image', 'evidence_overlay')",
            name=op.f("ck_artifacts_artifact_type"),
        ),
        sa.ForeignKeyConstraint(
            ["processing_run_id"],
            ["processing_runs.id"],
            name="fk_artifacts_processing_run_id_processing_runs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_artifacts"),
    )
    op.create_index("ix_artifacts_processing_run_id", "artifacts", ["processing_run_id"])
    op.create_table(
        "query_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pipeline_type", sa.String(length=32), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("answer_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("insufficient_evidence", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("retrieval_duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("generation_duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "pipeline_type IN ('baseline', 'docling_standard', 'docling_granite_vlm')",
            name=op.f("ck_query_runs_pipeline_type"),
        ),
        sa.CheckConstraint("top_k >= 1", name=op.f("ck_query_runs_top_k")),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], name="fk_query_runs_document_id_documents", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_query_runs"),
    )
    op.create_index("ix_query_runs_document_id", "query_runs", ["document_id"])
    op.create_table(
        "retrieval_hits",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("distance", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("rank >= 1", name=op.f("ck_retrieval_hits_rank")),
        sa.ForeignKeyConstraint(
            ["chunk_id"], ["chunks.id"], name="fk_retrieval_hits_chunk_id_chunks", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["query_run_id"],
            ["query_runs.id"],
            name="fk_retrieval_hits_query_run_id_query_runs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_retrieval_hits"),
        sa.UniqueConstraint("query_run_id", "rank", name="query_rank"),
    )
    op.create_index("ix_retrieval_hits_chunk_id", "retrieval_hits", ["chunk_id"])
    op.create_index("ix_retrieval_hits_query_run_id", "retrieval_hits", ["query_run_id"])
    op.create_table(
        "evaluation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("results_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("summary_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_evaluation_runs_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_evaluation_runs"),
    )
    op.create_index("ix_evaluation_runs_document_id", "evaluation_runs", ["document_id"])


def downgrade() -> None:
    """Remove all initial tables in reverse dependency order."""

    op.drop_table("evaluation_runs")
    op.drop_table("retrieval_hits")
    op.drop_table("query_runs")
    op.drop_table("artifacts")
    op.drop_table("provenance_records")
    op.drop_table("chunks")
    op.drop_table("processing_runs")
    op.drop_table("documents")
