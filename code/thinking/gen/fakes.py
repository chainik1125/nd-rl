"""An in-memory stand-in for the Message Batches endpoints, used as `HttpBatchClient`'s transport
in tests: it checks the request headers and body shape and answers each request with a
`responder(params) -> result` (by default the pilot traces as structured JSON)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import dsl
from .client import ANTHROPIC_VERSION

Responder = Callable[[dict[str, Any], str], dict[str, Any]]


def pilot_selection(k12: Path) -> list[dict[str, Any]]:
    """Selection records for the pilot's K12 rows (ids match the pilot trace ids)."""
    want = {int(k.split("_")[1]): k for k in dsl.pilot_records() if k.startswith("k12_")}
    out = []
    with open(k12) as f:
        for i, line in enumerate(f):
            if i in want:
                r = json.loads(line)
                out.append(
                    {
                        "id": want[i],
                        "source": f"K12 row {i}",
                        "row": i,
                        "prompt": r["prompt"],
                        "proof": r["proof"],
                    }
                )
            if i >= max(want):
                break
    return out


def message(
    model: str, text: str, stop: str = "end_turn", usage: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "type": "succeeded",
        "message": {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig_fake"},
                {"type": "text", "text": text},
            ],
            "stop_reason": stop,
            "stop_details": None,
            "usage": usage
            or {
                "input_tokens": 1200,
                "output_tokens": 900,
                "cache_read_input_tokens": 3000,
                "cache_creation_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
        },
    }


def pilot_responder(broken: dict[str, str] | None = None) -> Responder:
    """Answers with the pilot trace of the request's proof.  `broken[proof_id]` damages it:
    `false_look` / `always_bad` (a false look at the first decision step; `false_look` is fixed on
    the follow-up turn), `truncated` (stop_reason max_tokens on the first attempt), `errored`
    (an overloaded error on the first attempt), `refusal`."""
    pilot = dsl.pilot_records()
    broken = broken or {}

    def respond(params: dict[str, Any], custom_id: str) -> dict[str, Any]:
        pid = custom_id.rsplit("-", 2)[0]
        rec = pilot[pid]
        first = custom_id.endswith("1")
        followup = len(params["messages"]) > 1
        kind = broken.get(pid)
        steps = [dict(s) for s in rec["steps"]]
        if kind == "errored" and first:
            return {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "busy"},
                },
            }
        if kind == "refusal":
            r = message(params["model"], "", stop="refusal")
            r["message"]["stop_details"] = {"type": "refusal", "category": "cyber"}
            return r
        if kind == "truncated" and first:
            return message(params["model"], json.dumps({"steps": steps})[:200], stop="max_tokens")
        if kind == "always_bad" or (kind == "false_look" and not followup):
            k = next(i for i, s in enumerate(steps) if s["think"])
            steps[k]["think"] = "n1 : ( P & Q )\n" + steps[k]["think"]
        return message(params["model"], json.dumps({"steps": steps}, ensure_ascii=False))

    return respond


class FakeBatchAPI:
    """Transport: (method, url, headers, body, timeout) -> (status, bytes)."""

    def __init__(
        self,
        responder: Responder,
        expect_key: str | None = None,
        base: str = "https://api.anthropic.com",
    ):
        self.responder = responder
        self.expect_key = expect_key
        self.base = base
        self.batches: dict[str, dict[str, Any]] = {}
        self.results: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[tuple[str, str]] = []
        self.bodies: list[dict[str, Any]] = []

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> tuple[int, bytes]:
        self.calls.append((method, url))
        if self.expect_key is not None and headers.get("x-api-key") != self.expect_key:
            return 401, json.dumps(
                {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}
            ).encode()
        if headers.get("anthropic-version") != ANTHROPIC_VERSION:
            return (
                400,
                b'{"error": {"type": "invalid_request_error", "message": "anthropic-version"}}',
            )
        path = url[len(self.base) :]
        if method == "POST" and path == "/v1/messages/batches":
            req = json.loads(body or b"{}")
            self.bodies.append(req)
            bid = f"msgbatch_fake{len(self.batches) + 1:04d}"
            res = [
                {"custom_id": r["custom_id"], "result": self.responder(r["params"], r["custom_id"])}
                for r in req["requests"]
            ]
            self.results[bid] = res
            counts = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
            for r in res:
                counts[r["result"]["type"]] += 1
            self.batches[bid] = {
                "id": bid,
                "type": "message_batch",
                "processing_status": "in_progress",
                "request_counts": {**dict.fromkeys(counts, 0), "processing": len(res)},
                "results_url": None,
                "_final": counts,
            }
            return 200, json.dumps(
                {k: v for k, v in self.batches[bid].items() if k != "_final"}
            ).encode()
        if (
            method == "GET"
            and path.startswith("/v1/messages/batches/")
            and path.endswith("/results")
        ):
            bid = path.split("/")[-2]
            return 200, "\n".join(json.dumps(r) for r in self.results[bid]).encode()
        if method == "GET" and path.startswith("/v1/messages/batches/"):
            bid = path.split("/")[-1]
            b = self.batches[bid]
            b.update(
                processing_status="ended",
                request_counts=b["_final"],
                results_url=f"{self.base}/v1/messages/batches/{bid}/results",
            )
            return 200, json.dumps({k: v for k, v in b.items() if k != "_final"}).encode()
        return 404, b'{"error": {"type": "not_found_error", "message": "?"}}'
