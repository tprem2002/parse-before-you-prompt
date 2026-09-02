"""Enumerations shared by configuration and persistence models."""

from enum import StrEnum


class AzureOpenAIAuthMode(StrEnum):
    """Supported Azure OpenAI authentication modes."""

    API_KEY = "api_key"
    ENTRA = "entra"


class PipelineType(StrEnum):
    """Document processing paths."""

    BASELINE = "baseline"
    DOCLING_STANDARD = "docling_standard"
    DOCLING_GRANITE_VLM = "docling_granite_vlm"


class ProcessingStatus(StrEnum):
    """Lifecycle states for document processing."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class IndexMode(StrEnum):
    """Whether a processing worker may call Azure and Chroma."""

    AUTO = "auto"
    REQUIRED = "required"
    SKIP = "skip"


class ChunkRole(StrEnum):
    """How a stored chunk is used."""

    HIERARCHICAL_INSPECTION = "hierarchical_inspection"
    VECTOR_INDEX = "vector_index"


class ChunkKind(StrEnum):
    """Document content represented by a chunk."""

    TEXT = "text"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    PICTURE = "picture"
    FORMULA = "formula"
    CODE = "code"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORM = "form"
    KEY_VALUE = "key_value"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ContentClassification(StrEnum):
    """Whether chunk text is source content, derived content, or both."""

    SOURCE = "source"
    DERIVED = "derived"
    MIXED = "mixed"


class HeaderRepetitionStatus(StrEnum):
    """Observed table-header repetition outcome for a persisted chunk."""

    REPEATED = "repeated"
    NOT_REPEATED = "not_repeated"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ProvenanceRole(StrEnum):
    """How a stored source region supports its chunk text."""

    DIRECT_SOURCE_TEXT = "direct_source_text"
    SOURCE_IMAGE_REGION = "source_image_region"
    DERIVED_VISUAL_ANCHOR = "derived_visual_anchor"


class ArtifactType(StrEnum):
    """Filesystem artifacts recorded for a processing run."""

    ORIGINAL_PDF = "original_pdf"
    DOCLING_JSON = "docling_json"
    MARKDOWN = "markdown"
    BASELINE_TEXT = "baseline_text"
    PAGE_IMAGE = "page_image"
    PICTURE_IMAGE = "picture_image"
    TABLE_IMAGE = "table_image"
    CONVERSION_MANIFEST = "conversion_manifest"
    EVIDENCE_OVERLAY = "evidence_overlay"
