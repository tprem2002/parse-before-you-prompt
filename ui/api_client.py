"""Typed FastAPI client used by every Streamlit page.

The UI deliberately knows only the public HTTP contract. It never imports backend
services, opens PostgreSQL/Chroma, parses PDFs, or calls Azure OpenAI directly.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import httpx


JsonObject = dict[str, Any]
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@dataclass(frozen=True, slots=True)
class BinaryPayload:
    """A safe binary API response with display-relevant metadata only."""

    content: bytes
    content_type: str
    filename: str | None
    request_id: str
    headers: Mapping[str, str]


class ApiClientError(RuntimeError):
    """Sanitized API failure suitable for direct UI presentation."""

    def __init__(
        self,
        *,
        title: str,
        message: str,
        code: str,
        request_id: str,
        status_code: int | None = None,
        details: JsonObject | None = None,
    ) -> None:
        super().__init__(message)
        self.title = title
        self.message = message
        self.code = code
        self.request_id = request_id
        self.status_code = status_code
        self.details = details or {}


def _configured_base_url() -> str:
    """Read one non-secret process setting without loading an env file."""

    value = os.environ.get("PBTP_API_BASE_URL") or os.environ.get("UI_API_BASE_URL")
    return (value or "http://127.0.0.1:8080").rstrip("/")


def _filename_from_headers(headers: httpx.Headers) -> str | None:
    disposition = headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)', disposition, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip().replace("\\", "_").replace("/", "_")
    return value[:200] or None


class ApiClient:
    """Reusable synchronous client for the public FastAPI surface."""

    def __init__(self, base_url: str | None = None) -> None:
        self._client = httpx.Client(
            base_url=(base_url or _configured_base_url()).rstrip("/"),
            timeout=httpx.Timeout(connect=3.0, read=130.0, write=30.0, pool=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=False,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        request_id: str | None = None,
        accepted_statuses: set[int] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        correlation_id = (
            request_id
            if request_id and _SAFE_REQUEST_ID.fullmatch(request_id)
            else str(uuid.uuid4())
        )
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["X-Request-ID"] = correlation_id
        try:
            response = self._client.request(method, path, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiClientError(
                title="The API did not respond in time",
                message=(
                    "The request timed out. The operation may still be running; "
                    "refresh its status before trying again."
                ),
                code="api_timeout",
                request_id=correlation_id,
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            raise ApiClientError(
                title="API unavailable",
                message=(
                    "The local FastAPI service could not be reached. "
                    "Start it, then refresh status."
                ),
                code="api_unavailable",
                request_id=correlation_id,
                status_code=None,
            ) from exc

        if response.is_success or response.status_code in (accepted_statuses or set()):
            return response
        self._raise_for_error(response, fallback_request_id=correlation_id)
        raise AssertionError("unreachable")

    @staticmethod
    def _raise_for_error(response: httpx.Response, *, fallback_request_id: str) -> None:
        request_id = response.headers.get("X-Request-ID") or fallback_request_id
        code = f"http_{response.status_code}"
        message = "The API could not complete the request."
        details: JsonObject = {}
        try:
            payload = response.json()
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
            request_id = str(error.get("request_id") or request_id)
            raw_details = error.get("details")
            if isinstance(raw_details, dict):
                details = raw_details
        titles = {
            404: "Resource unavailable",
            409: "Request conflicts with current state",
            422: "Request needs attention",
            501: "Feature not available yet",
            502: "Model response could not be validated",
            503: "Required service unavailable",
            504: "Provider timeout",
        }
        raise ApiClientError(
            title=titles.get(response.status_code, "Request failed"),
            message=message,
            code=code,
            request_id=request_id,
            status_code=response.status_code,
            details=details,
        )

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            request_id = response.headers.get("X-Request-ID", "unavailable")
            raise ApiClientError(
                title="Unexpected API response",
                message="The API returned content that did not match its JSON contract.",
                code="invalid_api_response",
                request_id=request_id,
                status_code=response.status_code,
            ) from exc

    def health(self) -> JsonObject:
        response = self._request("GET", "/health", accepted_statuses={503})
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def configuration_status(self) -> JsonObject:
        return dict(self._json("GET", "/configuration/status"))

    def list_documents(self, *, limit: int = 100, offset: int = 0) -> list[JsonObject]:
        return list(
            self._json("GET", "/documents", params={"limit": limit, "offset": offset})
        )

    def get_document(self, document_id: str) -> JsonObject:
        return dict(self._json("GET", f"/documents/{document_id}"))

    def upload_document(self, *, filename: str, content: bytes) -> JsonObject:
        files = {"file": (filename, content, "application/pdf")}
        return dict(self._json("POST", "/documents/upload", files=files))

    def start_processing(
        self,
        document_id: str,
        *,
        pipeline_type: str,
        index_mode: str,
        force_reprocess: bool = False,
        retry_failed: bool = False,
    ) -> JsonObject:
        return dict(
            self._json(
                "POST",
                f"/documents/{document_id}/process",
                json={
                    "pipeline_type": pipeline_type,
                    "index_mode": index_mode,
                    "force_reprocess": force_reprocess,
                    "retry_failed": retry_failed,
                },
            )
        )

    def get_processing_run(self, run_id: str) -> JsonObject:
        return dict(self._json("GET", f"/processing-runs/{run_id}"))

    def list_artifacts(self, run_id: str) -> list[JsonObject]:
        return list(self._json("GET", f"/processing-runs/{run_id}/artifacts"))

    def get_artifact_content(self, artifact_id: str) -> BinaryPayload:
        return self._binary("GET", f"/artifacts/{artifact_id}/content")

    def list_chunks(
        self,
        run_id: str,
        *,
        chunk_role: str | None = None,
        kind: str | None = None,
        page: int | None = None,
        offset: int = 0,
        limit: int = 100,
        include_text: bool = False,
    ) -> JsonObject:
        params: dict[str, str | int | bool] = {
            "offset": offset,
            "limit": limit,
            "include_text": include_text,
        }
        if chunk_role:
            params["chunk_role"] = chunk_role
        if kind:
            params["kind"] = kind
        if page:
            params["page"] = page
        return dict(
            self._json("GET", f"/processing-runs/{run_id}/chunks", params=params)
        )

    def get_chunk(self, chunk_id: str) -> JsonObject:
        return dict(self._json("GET", f"/chunks/{chunk_id}"))

    def get_evidence_image(
        self, chunk_id: str, *, page_no: int | None = None
    ) -> BinaryPayload:
        params = {"page_no": page_no} if page_no is not None else None
        return self._binary("GET", f"/chunks/{chunk_id}/evidence-image", params=params)

    def run_rag_query(
        self, *, processing_run_id: str, question: str, top_k: int
    ) -> JsonObject:
        return dict(
            self._json(
                "POST",
                "/rag/query",
                json={
                    "processing_run_id": processing_run_id,
                    "question": question,
                    "top_k": top_k,
                },
            )
        )

    def start_evaluation(
        self,
        *,
        document_id: str,
        baseline_processing_run_id: str,
        docling_processing_run_id: str,
        top_k: int = 5,
        force_new: bool = False,
    ) -> JsonObject:
        return dict(
            self._json(
                "POST",
                "/evaluation/run",
                json={
                    "document_id": document_id,
                    "baseline_processing_run_id": baseline_processing_run_id,
                    "docling_processing_run_id": docling_processing_run_id,
                    "top_k": top_k,
                    "force_new": force_new,
                    "execute": True,
                    "confirmation": "RUN_PROJECT_AURORA_EVALUATION",
                },
            )
        )

    def evaluation_configuration(
        self,
        *,
        document_id: str,
        baseline_processing_run_id: str,
        docling_processing_run_id: str,
        top_k: int = 5,
    ) -> JsonObject:
        return dict(
            self._json(
                "GET",
                "/evaluation/configuration",
                params={
                    "document_id": document_id,
                    "baseline_processing_run_id": baseline_processing_run_id,
                    "docling_processing_run_id": docling_processing_run_id,
                    "top_k": top_k,
                },
            )
        )

    def get_evaluation(
        self, evaluation_id: str, *, include_results: bool = False
    ) -> JsonObject:
        return dict(
            self._json(
                "GET",
                f"/evaluation/{evaluation_id}",
                params={"include_results": include_results},
            )
        )

    def export_evaluation_csv(self, evaluation_id: str) -> BinaryPayload:
        return self._binary("GET", f"/evaluation/{evaluation_id}/export.csv")

    def export_evaluation_json(self, evaluation_id: str) -> BinaryPayload:
        return self._binary("GET", f"/evaluation/{evaluation_id}/export.json")

    def _binary(self, method: str, path: str, **kwargs: Any) -> BinaryPayload:
        response = self._request(method, path, headers={"Accept": "*/*"}, **kwargs)
        return BinaryPayload(
            content=response.content,
            content_type=response.headers.get(
                "content-type", "application/octet-stream"
            ).split(";", 1)[0],
            filename=_filename_from_headers(response.headers),
            request_id=response.headers.get("X-Request-ID", "unavailable"),
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower()
                in {"x-overlay-artifact-id", "x-rectangle-count", "x-overlay-reused"}
            },
        )


@lru_cache(maxsize=1)
def get_api_client() -> ApiClient:
    """Return one safe reusable client per Streamlit process."""

    return ApiClient()
