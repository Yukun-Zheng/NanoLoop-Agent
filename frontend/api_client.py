"""Typed, Streamlit-independent REST client for the NanoLoop backend."""

from __future__ import annotations

import json
import math
import mimetypes
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from typing import Any, BinaryIO, Generic, Literal, TypeAlias, TypeVar, cast
from urllib.parse import SplitResult, quote, urljoin, urlsplit
from uuid import uuid4

import httpx

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
ApiStatus: TypeAlias = Literal["success", "accepted"]
UploadContent: TypeAlias = bytes | BinaryIO

T = TypeVar("T")

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


@dataclass(frozen=True, slots=True)
class UploadPart:
    """Caller-owned upload content; this client never opens a filesystem path."""

    filename: str
    content: UploadContent
    content_type: str | None = None

    def resolved_content_type(self) -> str:
        guessed = mimetypes.guess_type(self.filename)[0]
        return self.content_type or guessed or "application/octet-stream"


@dataclass(frozen=True, slots=True)
class ApiResult(Generic[T]):
    data: T
    request_id: str
    status: ApiStatus


@dataclass(frozen=True, slots=True)
class ArtifactDownload:
    content: bytes
    filename: str | None
    content_type: str
    request_id: str


class ApiClientError(RuntimeError):
    """Backend, protocol, or transport failure with a correlation identifier."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        request_id: str,
        details: JsonObject | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.details = details or {}
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds

    def __str__(self) -> str:
        return (
            f"{self.code} (HTTP {self.status_code}, request_id={self.request_id}): {self.message}"
        )


class NanoLoopApiClient:
    """Synchronous REST facade suitable for Streamlit and other Python frontends."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        api_prefix: str = "/api/v1",
        timeout: float | httpx.Timeout | None = None,
        upload_timeout: float | httpx.Timeout | None = None,
        client: httpx.Client | None = None,
        request_id_factory: Callable[[], str] | None = None,
        max_retries: int = 2,
        default_retry_delay: float = 3.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) origin/path without credentials or query")
        normalized_prefix = "/" + api_prefix.strip("/")
        self._base_url = base_url.rstrip("/")
        self._api_prefix = normalized_prefix
        self._origin = _origin(parsed)
        self._api_key = _normalize_api_key(api_key)
        self._timeout = timeout or httpx.Timeout(30.0, connect=5.0, pool=5.0)
        self._upload_timeout = upload_timeout or httpx.Timeout(
            300.0,
            connect=5.0,
            pool=5.0,
        )
        self._request_id_factory = request_id_factory or (lambda: f"web_{uuid4().hex}")
        self._max_retries = max(0, max_retries)
        self._default_retry_delay = max(0.0, default_retry_delay)
        self._owns_client = client is None
        self._client = client or httpx.Client(follow_redirects=False)

    def __enter__(self) -> NanoLoopApiClient:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def health(self) -> ApiResult[JsonObject]:
        return self._request_json("GET", "/health")

    def list_models(
        self,
        *,
        status: str | None = None,
        family: str | None = None,
        variant: str | None = None,
        quality_tier: str | None = None,
        material: str | None = None,
    ) -> ApiResult[JsonObject]:
        params = httpx.QueryParams(
            {
                key: value
                for key, value in {
                    "status": status,
                    "family": family,
                    "variant": variant,
                    "quality_tier": quality_tier,
                    "material": material,
                }.items()
                if value is not None
            }
        )
        return self._request_json("GET", "/models", params=params)

    def recommend_models(self, payload: Mapping[str, Any]) -> ApiResult[JsonObject]:
        return self._request_json("POST", "/models/recommend", json_body=payload)

    def create_analysis(
        self,
        files: Sequence[UploadPart],
        metadata: Mapping[str, Any],
        *,
        timeout: float | httpx.Timeout | None = None,
    ) -> ApiResult[JsonObject]:
        if not files:
            raise ValueError("create_analysis requires at least one upload")
        _validate_uploads(files)
        multipart = [
            (
                "files",
                (part.filename, part.content, part.resolved_content_type()),
            )
            for part in files
        ]
        return self._request_json(
            "POST",
            "/analyses",
            files=multipart,
            form={"metadata_json": _encode_json(metadata)},
            timeout=timeout or self._upload_timeout,
        )

    def get_analysis(self, job_id: str) -> ApiResult[JsonObject]:
        return self._request_json("GET", f"/analyses/{_segment(job_id, 'job_id')}")

    def get_boxes(self, job_id: str, image_id: str) -> ApiResult[JsonObject]:
        return self._request_json(
            "GET",
            self._boxes_path(job_id, image_id),
        )

    def replace_boxes(
        self,
        job_id: str,
        image_id: str,
        *,
        expected_revision: int,
        boxes: Sequence[Mapping[str, Any]],
    ) -> ApiResult[JsonObject]:
        return self._request_json(
            "PUT",
            self._boxes_path(job_id, image_id),
            json_body={
                "expected_revision": expected_revision,
                "boxes": [dict(box) for box in boxes],
            },
        )

    def create_runs(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> ApiResult[JsonObject]:
        return self._request_json(
            "POST",
            f"/analyses/{_segment(job_id, 'job_id')}/runs",
            json_body=payload,
        )

    def get_run(self, run_id: str) -> ApiResult[JsonObject]:
        return self._request_json("GET", f"/runs/{_segment(run_id, 'run_id')}")

    def upload_corrected_mask(
        self,
        run_id: str,
        file: UploadPart,
    ) -> ApiResult[JsonObject]:
        _validate_uploads([file])
        return self._request_json(
            "POST",
            f"/runs/{_segment(run_id, 'run_id')}/corrected-mask",
            files=[
                (
                    "file",
                    (file.filename, file.content, file.resolved_content_type()),
                )
            ],
            timeout=self._upload_timeout,
        )

    def review_run(
        self,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> ApiResult[JsonObject]:
        return self._request_json(
            "POST",
            f"/runs/{_segment(run_id, 'run_id')}/review",
            json_body=payload,
        )

    def query_analysis(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> ApiResult[JsonObject]:
        return self._request_json(
            "POST",
            f"/analyses/{_segment(job_id, 'job_id')}/query",
            json_body=payload,
        )

    def ingest_knowledge_document(
        self,
        file: UploadPart,
        metadata: Mapping[str, Any],
    ) -> ApiResult[JsonObject]:
        _validate_uploads([file])
        return self._request_json(
            "POST",
            "/knowledge/documents",
            files=[
                (
                    "file",
                    (file.filename, file.content, file.resolved_content_type()),
                )
            ],
            form={"metadata_json": _encode_json(metadata)},
            timeout=self._upload_timeout,
        )

    def list_knowledge_documents(self) -> ApiResult[JsonObject]:
        return self._request_json("GET", "/knowledge/documents")

    def update_knowledge_document(
        self,
        doc_id: str,
        *,
        enabled: bool,
    ) -> ApiResult[JsonObject]:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        return self._request_json(
            "PATCH",
            f"/knowledge/documents/{_segment(doc_id, 'doc_id')}",
            json_body={"enabled": enabled},
        )

    def reindex_knowledge(self, *, force: bool = False) -> ApiResult[JsonObject]:
        return self._request_json(
            "POST",
            "/knowledge/reindex",
            json_body={"force": force},
        )

    def export_analysis(
        self,
        job_id: str,
        *,
        run_ids: Sequence[str] | None = None,
    ) -> ApiResult[JsonObject]:
        params = None
        if run_ids:
            params = httpx.QueryParams([("run_ids", run_id) for run_id in dict.fromkeys(run_ids)])
        return self._request_json(
            "GET",
            f"/analyses/{_segment(job_id, 'job_id')}/export",
            params=params,
        )

    def download_artifact(self, download_url: str) -> ArtifactDownload:
        url = self._validated_download_url(download_url)
        outbound_request_id = self._new_request_id()
        response = self._send(
            "GET",
            url,
            request_id=outbound_request_id,
            headers={"Accept": "application/octet-stream"},
            timeout=self._upload_timeout,
        )
        if not response.is_success:
            self._raise_response_error(response, outbound_request_id)
        response_request_id = _response_request_id(response, outbound_request_id)
        content_type = response.headers.get("content-type", "application/octet-stream")
        return ArtifactDownload(
            content=response.content,
            filename=_response_filename(response),
            content_type=content_type.split(";", maxsplit=1)[0].strip(),
            request_id=response_request_id,
        )

    def _boxes_path(self, job_id: str, image_id: str) -> str:
        return (
            f"/analyses/{_segment(job_id, 'job_id')}/images/{_segment(image_id, 'image_id')}/boxes"
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: httpx.QueryParams | None = None,
        json_body: Mapping[str, Any] | None = None,
        files: list[tuple[str, tuple[str, UploadContent, str]]] | None = None,
        form: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> ApiResult[JsonObject]:
        resolved_timeout = timeout or self._timeout
        retry_after_seconds: float | None = None
        for attempt in range(self._max_retries + 1):
            outbound_request_id = self._new_request_id()
            response = self._send(
                method,
                self._api_url(path),
                request_id=outbound_request_id,
                params=params,
                json_body=json_body,
                files=files,
                form=form,
                timeout=resolved_timeout,
            )
            if (
                response.status_code == 429
                and attempt < self._max_retries
            ):
                retry_after_seconds = _parse_retry_after(
                    response.headers.get("retry-after"),
                    default=self._default_retry_delay,
                )
                time.sleep(retry_after_seconds)
                continue
            if not response.is_success:
                self._raise_response_error(
                    response,
                    outbound_request_id,
                    retry_after_seconds=retry_after_seconds,
                )
            payload = _decode_json_response(response, outbound_request_id)
            request_id = _envelope_request_id(payload, response, outbound_request_id)
            status = payload.get("status")
            if status == "error":
                self._raise_envelope_error(response.status_code, payload, request_id)
            if status not in {"success", "accepted"}:
                raise _protocol_error(
                    response.status_code,
                    request_id,
                    "response envelope has an invalid status",
                )
            data = payload.get("data")
            if not isinstance(data, dict):
                raise _protocol_error(
                    response.status_code,
                    request_id,
                    "response envelope data must be an object",
                )
            return ApiResult(
                data=_json_object(data),
                request_id=request_id,
                status=cast(ApiStatus, status),
            )
        # Unreachable: the final iteration always returns or raises.
        raise RuntimeError("retry loop exhausted without resolution")  # pragma: no cover

    def _send(
        self,
        method: str,
        url: str,
        *,
        request_id: str,
        headers: Mapping[str, str] | None = None,
        params: httpx.QueryParams | None = None,
        json_body: Mapping[str, Any] | None = None,
        files: list[tuple[str, tuple[str, UploadContent, str]]] | None = None,
        form: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout,
    ) -> httpx.Response:
        request_headers = {
            "Accept": "application/json",
            "User-Agent": "NanoLoop-Frontend/0.1",
            "X-Request-ID": request_id,
        }
        request_headers.update(
            {
                name: value
                for name, value in (headers or {}).items()
                if name.casefold() != "x-api-key"
            }
        )
        if self._api_key is not None:
            request_headers["X-API-Key"] = self._api_key
        try:
            return self._client.request(
                method,
                url,
                headers=request_headers,
                params=params,
                json=dict(json_body) if json_body is not None else None,
                files=files,
                data=form,
                timeout=timeout,
            )
        except httpx.TimeoutException as error:
            raise ApiClientError(
                status_code=0,
                code="REQUEST_TIMEOUT",
                message="请求后端超时",
                request_id=request_id,
                details={"error_type": type(error).__name__},
                retryable=True,
            ) from error
        except httpx.RequestError as error:
            raise ApiClientError(
                status_code=0,
                code="TRANSPORT_ERROR",
                message="无法连接后端服务",
                request_id=request_id,
                details={"error_type": type(error).__name__},
                retryable=True,
            ) from error

    def _raise_response_error(
        self,
        response: httpx.Response,
        outbound_request_id: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        try:
            payload = _decode_json_response(response, outbound_request_id)
        except ApiClientError:
            request_id = _response_request_id(response, outbound_request_id)
            raise ApiClientError(
                status_code=response.status_code,
                code=f"HTTP_{response.status_code}",
                message="后端返回了非 JSON 错误响应",
                request_id=request_id,
                retryable=response.status_code == 429 or response.status_code >= 500,
                retry_after_seconds=retry_after_seconds,
            ) from None
        request_id = _envelope_request_id(payload, response, outbound_request_id)
        self._raise_envelope_error(
            response.status_code,
            payload,
            request_id,
            retry_after_seconds=retry_after_seconds,
        )

    @staticmethod
    def _raise_envelope_error(
        status_code: int,
        payload: JsonObject,
        request_id: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        error = payload.get("error")
        if not isinstance(error, dict):
            raise _protocol_error(
                status_code,
                request_id,
                "error response is missing its error object",
            )
        code = error.get("code")
        message = error.get("message")
        details = error.get("details")
        retryable = error.get("retryable", False)
        if not isinstance(code, str) or not isinstance(message, str):
            raise _protocol_error(
                status_code,
                request_id,
                "error response has invalid code or message",
            )
        raise ApiClientError(
            status_code=status_code,
            code=code,
            message=message,
            request_id=request_id,
            details=details if isinstance(details, dict) else {},
            retryable=retryable if isinstance(retryable, bool) else False,
            retry_after_seconds=retry_after_seconds,
        )

    def _api_url(self, path: str) -> str:
        return f"{self._base_url}{self._api_prefix}{path}"

    def _validated_download_url(self, download_url: str) -> str:
        if not download_url or any(character.isspace() for character in download_url):
            raise ValueError("download_url must be a non-empty URL without whitespace")
        candidate = urljoin(f"{self._base_url}/", download_url)
        parsed = urlsplit(candidate)
        if _origin(parsed) != self._origin:
            raise ValueError("signed artifact URLs must use the configured backend origin")
        expected_prefix = f"{self._api_prefix}/files/"
        if not parsed.path.startswith(expected_prefix) or parsed.fragment:
            raise ValueError("download_url is not a signed NanoLoop artifact URL")
        return candidate

    def _new_request_id(self) -> str:
        request_id = self._request_id_factory()
        if not _REQUEST_ID_PATTERN.fullmatch(request_id):
            raise ValueError("request_id_factory returned an invalid correlation ID")
        return request_id


def _normalize_api_key(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError("api_key must be a string or None")
    if value != value.strip() or any(not 0x21 <= ord(character) <= 0x7E for character in value):
        raise ValueError("api_key must contain only visible ASCII characters without whitespace")
    return value


def _parse_retry_after(header_value: str | None, *, default: float) -> float:
    """Parse the ``Retry-After`` response header into a delay in seconds.

    The header may be either a non-negative integer (seconds) or an HTTP-date.
    For HTTP-date values or missing headers, fall back to *default*.
    """
    if header_value is None:
        return default
    try:
        seconds = float(header_value)
    except (ValueError, TypeError):
        return default
    return max(0.0, seconds)


def _validate_uploads(parts: Sequence[UploadPart]) -> None:
    filenames: set[str] = set()
    for part in parts:
        forbidden = ("/", "\\", "\x00")
        if not part.filename or any(character in part.filename for character in forbidden):
            raise ValueError("upload filenames must be non-empty single path components")
        if part.filename in filenames:
            raise ValueError(f"duplicate upload filename: {part.filename}")
        filenames.add(part.filename)


def _segment(value: str, field: str) -> str:
    if not value:
        raise ValueError(f"{field} cannot be empty")
    return quote(value, safe="")


def _encode_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not JSON serializable") from error


def _decode_json_response(response: httpx.Response, request_id: str) -> JsonObject:
    try:
        value = response.json()
        return _json_object(value)
    except (TypeError, ValueError) as error:
        raise _protocol_error(
            response.status_code,
            _response_request_id(response, request_id),
            "backend response is not a valid JSON object",
        ) from error


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return {str(key): _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise ValueError("unsupported JSON value")


def _envelope_request_id(
    payload: JsonObject,
    response: httpx.Response,
    fallback: str,
) -> str:
    body_request_id = payload.get("request_id")
    if not isinstance(body_request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(body_request_id):
        raise _protocol_error(
            response.status_code,
            _response_request_id(response, fallback),
            "response envelope has no valid request_id",
        )
    header_request_id = response.headers.get("x-request-id")
    if header_request_id is not None and header_request_id != body_request_id:
        raise _protocol_error(
            response.status_code,
            fallback,
            "response header and envelope request IDs do not match",
            details={
                "header_request_id": header_request_id,
                "body_request_id": body_request_id,
            },
        )
    return body_request_id


def _response_request_id(response: httpx.Response, fallback: str) -> str:
    value = cast(str | None, response.headers.get("x-request-id"))
    if value is not None and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return fallback


def _response_filename(response: httpx.Response) -> str | None:
    content_disposition = response.headers.get("content-disposition")
    if not content_disposition:
        return None
    message = Message()
    message["content-disposition"] = content_disposition
    filename = message.get_filename()
    return filename if isinstance(filename, str) and filename else None


def _protocol_error(
    status_code: int,
    request_id: str,
    message: str,
    *,
    details: JsonObject | None = None,
    retry_after_seconds: float | None = None,
) -> ApiClientError:
    return ApiClientError(
        status_code=status_code,
        code="API_PROTOCOL_ERROR",
        message=message,
        request_id=request_id,
        details=details,
        retryable=False,
        retry_after_seconds=retry_after_seconds,
    )


def _origin(parsed: SplitResult) -> tuple[str, str | None, int | None]:
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname, parsed.port or default_port
