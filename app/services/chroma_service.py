"""Reusable Chroma HTTP service for externally generated cosine vectors."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from app.core.config import Settings, get_settings
from app.core.errors import ExternalServiceError


class ChromaIndexError(ExternalServiceError):
    """Chroma indexing or collection validation failed."""


@dataclass(frozen=True, slots=True)
class ChromaRecord:
    """One externally embedded record ready for a flat Chroma upsert."""

    vector_id: str
    embedding: tuple[float, ...]
    document: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class ChromaCollectionHandle:
    """A validated collection and whether this operation created it."""

    collection: Collection
    created: bool


@dataclass(frozen=True, slots=True)
class ChromaStatus:
    """Safe connectivity details for configuration reporting."""

    reachable: bool
    collection_count: int | None


@dataclass(frozen=True, slots=True)
class ChromaQueryHit:
    """One ranked result returned exactly once from a vector query."""

    vector_id: str
    distance: float
    metadata: dict[str, str | int | float | bool]
    document: str | None


def collection_name_for(pipeline_type: str, embedding_fingerprint: str) -> str:
    """Return a stable pipeline-isolated name using a shortened identity hash."""

    prefixes = {
        "baseline": "pbtp_baseline",
        "docling_standard": "pbtp_docling_standard",
        "docling_granite_vlm": "pbtp_docling_vlm",
    }
    try:
        prefix = prefixes[pipeline_type]
    except KeyError as exc:
        raise ChromaIndexError(f"Unsupported Chroma pipeline type: {pipeline_type}") from exc
    return f"{prefix}_{embedding_fingerprint[:12].lower()}"


class ChromaService:
    """Lazy reusable server-backed Chroma client with no embedding function."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: chromadb.ClientAPI | None = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                self._client = chromadb.HttpClient(
                    host=self._settings.chroma_host,
                    port=self._settings.chroma_port,
                    ssl=self._settings.chroma_ssl,
                )
            return self._client

    def status(self) -> ChromaStatus:
        """Check connectivity and count collections without creating one."""

        try:
            client = self._get_client()
            client.heartbeat()
            return ChromaStatus(reachable=True, collection_count=client.count_collections())
        except Exception:
            return ChromaStatus(reachable=False, collection_count=None)

    def heartbeat(self) -> int:
        return self._get_client().heartbeat()

    def count_collections(self) -> int:
        return self._get_client().count_collections()

    def delete_demo_collections(self) -> tuple[str, ...]:
        """Delete only collections owned by this demonstration."""

        client = self._get_client()
        deleted: list[str] = []
        for item in client.list_collections():
            name = item.name if hasattr(item, "name") else str(item)
            if not name.startswith("pbtp_"):
                continue
            client.delete_collection(name=name)
            deleted.append(name)
        return tuple(sorted(deleted))

    def ensure_collection(
        self,
        *,
        pipeline_type: str,
        embedding_fingerprint: str,
        vector_dimension: int,
        input_representation_version: str,
        provider_name: str,
        embedding_provider_version: str,
        deployment_name: str,
        base_url_hash: str,
        provisional_fingerprint: str,
    ) -> ChromaCollectionHandle:
        """Create or validate one fingerprinted cosine collection."""

        client = self._get_client()
        name = collection_name_for(pipeline_type, embedding_fingerprint)
        expected_metadata: dict[str, str | int | float | bool] = {
            "pipeline_type": pipeline_type,
            "embedding_fingerprint": embedding_fingerprint,
            "vector_dimension": vector_dimension,
            "input_representation_version": input_representation_version,
            "embedding_provider": provider_name,
            "embedding_provider_version": embedding_provider_version,
            "embedding_deployment": deployment_name,
            "base_url_hash": base_url_hash,
            "provisional_fingerprint": provisional_fingerprint,
            "external_embeddings": True,
        }
        created = False
        try:
            collection = client.get_collection(name=name, embedding_function=None)
        except NotFoundError:
            collection = client.get_or_create_collection(
                name=name,
                metadata=expected_metadata,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
            )
            created = True
        self.validate_collection(
            collection,
            pipeline_type=pipeline_type,
            embedding_fingerprint=embedding_fingerprint,
            vector_dimension=vector_dimension,
            input_representation_version=input_representation_version,
            embedding_provider_version=embedding_provider_version,
        )
        return ChromaCollectionHandle(collection=collection, created=created)

    @staticmethod
    def validate_collection(
        collection: Collection,
        *,
        pipeline_type: str,
        embedding_fingerprint: str,
        vector_dimension: int,
        input_representation_version: str,
        embedding_provider_version: str,
    ) -> None:
        """Reject any existing collection whose semantic identity does not match."""

        metadata = collection.metadata or {}
        expected = {
            "pipeline_type": pipeline_type,
            "embedding_fingerprint": embedding_fingerprint,
            "vector_dimension": vector_dimension,
            "input_representation_version": input_representation_version,
            "embedding_provider_version": embedding_provider_version,
            "external_embeddings": True,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        configuration: dict[str, Any] = collection.configuration
        hnsw = configuration.get("hnsw") or {}
        if hnsw.get("space") != "cosine":
            mismatches.append("hnsw.space")
        if mismatches:
            raise ChromaIndexError(
                "Existing Chroma collection does not match required indexing metadata: "
                + ", ".join(sorted(mismatches))
            )

    def get_validated_collection(
        self,
        *,
        name: str,
        pipeline_type: str,
        embedding_fingerprint: str,
        vector_dimension: int,
        input_representation_version: str,
        embedding_provider_version: str,
    ) -> Collection:
        """Retrieve an existing collection without allowing an embedding function."""

        try:
            collection = self._get_client().get_collection(name=name, embedding_function=None)
        except NotFoundError as exc:
            raise ChromaIndexError(f"Chroma collection not found: {name}") from exc
        self.validate_collection(
            collection,
            pipeline_type=pipeline_type,
            embedding_fingerprint=embedding_fingerprint,
            vector_dimension=vector_dimension,
            input_representation_version=input_representation_version,
            embedding_provider_version=embedding_provider_version,
        )
        return collection

    def existing_ids(self, collection: Collection, vector_ids: list[str]) -> set[str]:
        """Return the requested deterministic IDs that currently exist."""

        if not vector_ids:
            return set()
        result = collection.get(ids=vector_ids, include=[])
        return set(result["ids"])

    @staticmethod
    def query_by_vector(
        collection: Collection,
        *,
        query_vector: list[float],
        n_results: int,
        where: dict[str, Any],
    ) -> tuple[ChromaQueryHit, ...]:
        """Query one external vector and preserve Chroma's returned rank order."""

        if not query_vector:
            raise ChromaIndexError("Query vector is empty")
        if n_results < 1:
            raise ChromaIndexError("n_results must be at least one")
        try:
            result = collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            ids = result.get("ids")
            distances = result.get("distances")
            metadatas = result.get("metadatas")
            documents = result.get("documents")
            if not ids or distances is None or metadatas is None or documents is None:
                raise ChromaIndexError("Chroma returned an incomplete query result")
            if not all(len(value) == 1 for value in (ids, distances, metadatas, documents)):
                raise ChromaIndexError("Chroma returned an unexpected query batch shape")
            rows = (ids[0], distances[0], metadatas[0], documents[0])
            if len({len(value) for value in rows}) != 1:
                raise ChromaIndexError("Chroma returned inconsistent query result lengths")
            return tuple(
                ChromaQueryHit(
                    vector_id=str(vector_id),
                    distance=float(distance),
                    metadata=dict(metadata or {}),
                    document=document,
                )
                for vector_id, distance, metadata, document in zip(*rows, strict=True)
            )
        except ChromaIndexError:
            raise
        except Exception as exc:
            raise ChromaIndexError(
                f"Chroma vector query failed safely: {type(exc).__name__}"
            ) from None

    def upsert_records(self, collection: Collection, records: list[ChromaRecord]) -> None:
        """Upsert explicit vectors in backend-safe batches and verify every ID."""

        if not records:
            raise ChromaIndexError("No Chroma records were supplied")
        max_batch_size = self._get_client().get_max_batch_size()
        for start in range(0, len(records), max_batch_size):
            batch = records[start : start + max_batch_size]
            collection.upsert(
                ids=[record.vector_id for record in batch],
                embeddings=[list(record.embedding) for record in batch],
                documents=[record.document for record in batch],
                metadatas=[record.metadata for record in batch],
            )
        expected = {record.vector_id for record in records}
        actual = self.existing_ids(collection, sorted(expected))
        if actual != expected:
            raise ChromaIndexError(
                f"Chroma verification returned {len(actual)} of {len(expected)} vector IDs"
            )

    @staticmethod
    def delete_ids(collection: Collection, vector_ids: list[str]) -> None:
        """Best-effort cleanup for IDs known not to predate the operation."""

        if vector_ids:
            collection.delete(ids=vector_ids)

    def delete_collection_if_empty(self, name: str) -> bool:
        """Delete only an exact newly-created collection after cleanup left it empty."""

        try:
            collection = self._get_client().get_collection(name=name, embedding_function=None)
        except NotFoundError:
            return True
        if collection.count() != 0:
            return False
        self._get_client().delete_collection(name=name)
        return True


_service_lock = threading.Lock()
_service_instance: ChromaService | None = None
_service_key: tuple[str, int, bool] | None = None


def get_chroma_service(settings: Settings | None = None) -> ChromaService:
    """Return one shared HTTP service per non-secret Chroma configuration."""

    global _service_instance, _service_key
    configured_settings = settings or get_settings()
    key = (
        configured_settings.chroma_host,
        configured_settings.chroma_port,
        configured_settings.chroma_ssl,
    )
    with _service_lock:
        if _service_instance is None or _service_key != key:
            _service_instance = ChromaService(configured_settings)
            _service_key = key
        return _service_instance
