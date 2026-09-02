"""Allow conversion-manifest artifacts.

Revision ID: 0002_conversion_manifest
Revises: 0001_initial_schema
"""

from collections.abc import Sequence

from alembic import op


revision: str = "0002_conversion_manifest"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_OLD_VALUES = (
    "'original_pdf', 'docling_json', 'markdown', 'baseline_text', "
    "'page_image', 'picture_image', 'table_image', 'evidence_overlay'"
)
_NEW_VALUES = (
    "'original_pdf', 'docling_json', 'markdown', 'baseline_text', "
    "'page_image', 'picture_image', 'table_image', 'conversion_manifest', "
    "'evidence_overlay'"
)


def upgrade() -> None:
    """Extend the artifact type check constraint for conversion manifests."""

    op.drop_constraint(op.f("ck_artifacts_artifact_type"), "artifacts", type_="check")
    op.create_check_constraint(
        op.f("ck_artifacts_artifact_type"),
        "artifacts",
        f"artifact_type IN ({_NEW_VALUES})",
    )


def downgrade() -> None:
    """Restore the original artifact type constraint."""

    op.drop_constraint(op.f("ck_artifacts_artifact_type"), "artifacts", type_="check")
    op.create_check_constraint(
        op.f("ck_artifacts_artifact_type"),
        "artifacts",
        f"artifact_type IN ({_OLD_VALUES})",
    )
