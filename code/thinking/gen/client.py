"""A minimal Message Batches client over raw HTTPS (the `anthropic` SDK is not installed in the
shared venv).  The key is read from ANTHROPIC_API_KEY at call time and only ever placed in the
`x-api-key` request header: it is never printed, logged, stored on the object or written anywhere.

Endpoints (verified 2026-10-05, README.md): POST /v1/messages/batches, GET
/v1/messages/batches/{id}, GET the batch's `results_url` (JSONL).  Headers: x-api-key,
anthropic-version: 2023-06-01, content-type: application/json.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any, Protocol

API_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
KEY_ENV = "ANTHROPIC_API_KEY"
MAX_REQUESTS = 100_000
MAX_BYTES = 256 * 1024 * 1024

# (method, url, headers, body, timeout) -> (status, response bytes)
Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class BatchClient(Protocol):
    def create(self, requests: list[dict[str, Any]]) -> dict[str, Any]: ...

    def retrieve(self, batch_id: str) -> dict[str, Any]: ...

    def results(self, batch_id: str) -> Iterator[dict[str, Any]]: ...


def key_is_set(env: str = KEY_ENV) -> bool:
    return bool(os.environ.get(env))


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _error_text(raw: bytes) -> str:
    try:
        err = json.loads(raw).get("error", {})
        return f"{err.get('type', '?')}: {err.get('message', '')}"[:500]
    except (ValueError, AttributeError):
        return raw[:200].decode("utf-8", "replace")


class HttpBatchClient:
    def __init__(
        self,
        transport: Transport = urllib_transport,
        base_url: str = API_BASE,
        timeout: float = 300.0,
        key_env: str = KEY_ENV,
        get_retries: int = 4,
    ):
        self.transport = transport
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.key_env = key_env
        self.get_retries = get_retries

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.key_env)
        if not key:
            raise ApiError(0, f"{self.key_env} is not set")
        return {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _call(self, method: str, url: str, body: Any = None) -> bytes:
        if not url.startswith(self.base_url + "/"):
            raise ApiError(0, f"refusing to send the API key to {url.split('?')[0]!r}")
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        # a POST is never retried: an ambiguous failure could mean a batch was created (and billed)
        tries = 1 if method != "GET" else 1 + self.get_retries
        for k in range(tries):
            status, raw = self.transport(method, url, self._headers(), data, self.timeout)
            if 200 <= status < 300:
                return raw
            if (
                method != "GET"
                or status not in (408, 409, 429, 500, 502, 503, 504, 529)
                or k == tries - 1
            ):
                raise ApiError(status, _error_text(raw))
            time.sleep(min(60.0, 2.0**k))
        raise AssertionError("unreachable")

    def create(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        if not 0 < len(requests) <= MAX_REQUESTS:
            raise ApiError(0, f"a batch holds 1..{MAX_REQUESTS} requests, not {len(requests)}")
        body = {"requests": requests}
        if len(json.dumps(body).encode()) > MAX_BYTES:
            raise ApiError(0, "batch body exceeds 256 MB")
        return json.loads(self._call("POST", f"{self.base_url}/v1/messages/batches", body))

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        return json.loads(self._call("GET", f"{self.base_url}/v1/messages/batches/{batch_id}"))

    def results(self, batch_id: str) -> Iterator[dict[str, Any]]:
        b = self.retrieve(batch_id)
        url = b.get("results_url")
        if b.get("processing_status") != "ended" or not url:
            raise ApiError(0, f"batch {batch_id} has not ended ({b.get('processing_status')})")
        for line in self._call("GET", url).decode().splitlines():
            if line.strip():
                yield json.loads(line)
