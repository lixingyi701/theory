"""OpenAI-compatible HTTP transport and a durable, budgeted request journal."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .io import digest, read_json, write_json


class InfrastructureError(RuntimeError):
    pass


class CallBudgetExceeded(RuntimeError):
    pass


class UncertainRequest(InfrastructureError):
    pass


class HTTPTransport:
    def __init__(self, base_url: str, api_key: str, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.api_key, self.timeout = api_key, timeout

    def _request(self, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=body,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + self.api_key})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # No automatic retry: the server may already have generated/billed it.
            raise InfrastructureError(f"Model endpoint request failed ({type(exc).__name__})") from exc

    def health(self, expected_model: str) -> dict:
        data = self._request("/models")
        ids = [model["id"] for model in data.get("data", [])]
        if expected_model not in ids:
            raise InfrastructureError(f"Expected served model {expected_model!r}, got {ids!r}")
        return {"served_model": expected_model, "models": ids}

    def complete(self, payload: dict) -> dict:
        return self._request("/chat/completions", payload)


class CallJournal:
    """A logical call is cached; each actual dispatch counts, including uncertainty.

    On resume, cached responses can reconstruct an interrupted step. A dispatched
    request lacking a saved response stops by default, rather than silently
    sending it again. --retry-uncertain explicitly creates another billed attempt.
    """

    def __init__(self, directory: Path, transport, max_calls: int, retry_uncertain: bool = False):
        self.directory, self.transport = directory, transport
        self.max_calls, self.retry_uncertain = max_calls, retry_uncertain
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def dispatched(self) -> int:
        return len(list(self.directory.glob("*/attempt-*.request.json")))

    def complete(self, logical_id: str, payload: dict) -> dict:
        directory = self.directory / logical_id
        directory.mkdir(exist_ok=True)
        cached = directory / "response.json"
        if cached.exists():
            saved = read_json(cached)
            if saved["payload_sha256"] != digest(payload):
                raise InfrastructureError("Prompt/config changed while reusing a logical call")
            return saved
        attempts = sorted(directory.glob("attempt-*.request.json"))
        if attempts and not self.retry_uncertain:
            raise UncertainRequest(f"Unresolved request {logical_id}; inspect journal before --retry-uncertain")
        if self.dispatched >= self.max_calls:
            raise CallBudgetExceeded("Model-call budget exhausted")
        index = len(attempts)
        request_path = directory / f"attempt-{index:03d}.request.json"
        write_json(request_path, {"payload_sha256": digest(payload), "payload": payload,
                                  "dispatched_unix": time.time()})
        start = time.monotonic()
        response = self.transport.complete(payload)
        saved = {"payload_sha256": digest(payload), "response": response,
                 "latency_ms": (time.monotonic() - start) * 1000, "attempt": index}
        write_json(cached, saved)
        return saved

    def totals(self) -> dict:
        responses = [read_json(p) for p in self.directory.glob("*/response.json")]
        usage = [r["response"].get("usage") for r in responses]
        known = [u for u in usage if u is not None]
        validations = [read_json(p) for p in self.directory.glob("*/validation.json")]
        return {"dispatched_calls": self.dispatched, "saved_responses": len(responses),
                "uncertain_or_unrecorded_attempts": self.dispatched - len(responses),
                "usage_missing_responses": len(usage) - len(known),
                "format_invalid_responses": sum(not v["valid"] for v in validations),
                "truncated_responses": sum(v["truncated_output"] for v in validations),
                "input_tokens_known": sum(u.get("prompt_tokens", 0) for u in known),
                "output_tokens_known": sum(u.get("completion_tokens", 0) for u in known),
                "response_latency_ms": sum(r["latency_ms"] for r in responses)}


def response_content(saved: dict) -> tuple[str, str | None]:
    try:
        choice = saved["response"]["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Content is not text")
        return content, choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise InfrastructureError("Malformed model endpoint response") from exc


def endpoint_from_config(config: dict) -> HTTPTransport:
    return HTTPTransport(os.environ.get(config["api_base_env"], "http://127.0.0.1:8000/v1"),
                         os.environ.get(config["api_key_env"], "local-development"),
                         config["timeout_seconds"])
