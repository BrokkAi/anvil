"""Native, tool-free inference using Anvil's authentication and provider clients."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, TypedDict

from ._native import NativeClient, NativeError, __version__


class Message(TypedDict):
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    cached_read_tokens: int
    cached_write_tokens: int


@dataclass(frozen=True)
class InferenceResult:
    output: Any
    usage: Usage
    model: str
    reasoning_effort: str | None
    service_tier: str | None


class InferenceError(Exception):
    """Provider failure; ``kind`` is a stable machine-readable category."""
    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


class InvalidRequestError(InferenceError): pass
class AuthenticationError(InferenceError): pass
class RateLimitError(InferenceError): pass
class ContextLengthError(InferenceError): pass
class TransportError(InferenceError): pass
class StructuredOutputError(InferenceError): pass
class ProviderError(InferenceError): pass
class ClientClosedError(InferenceError): pass
class InferenceTimeoutError(InferenceError, TimeoutError): pass


_ERRORS = {
    "InvalidRequest": InvalidRequestError,
    "Authentication": AuthenticationError,
    "RateLimited": RateLimitError,
    "ContextLength": ContextLengthError,
    "Transport": TransportError,
    "StructuredOutput": StructuredOutputError,
    "Provider": ProviderError,
    "Cancelled": ClientClosedError,
    "Timeout": InferenceTimeoutError,
}


class Client:
    """Reusable async client. Use ``async with Client()`` or call ``close()``.

    Credentials are loaded lazily per provider and connections are reused.
    Cancelling an asyncio task cancels its native request. Closing the client
    cancels all active requests and permanently prevents subsequent requests.
    """
    def __init__(self) -> None:
        self._native = NativeClient()
        self._closed = False

    async def __aenter__(self) -> Client:
        if self._closed:
            raise ClientClosedError("Cancelled", "client closed")
        return self

    async def __aexit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True
        self._native.close()

    async def infer(
        self, *, model: str, messages: Sequence[Message], schema: Mapping[str, Any],
        schema_name: str = "response", reasoning_effort: str | None = None,
        service_tier: str | None = None, timeout: float = 120,
        idle_timeout: float = 60, stall_timeout: float = 30,
        validation_retries: int = 1,
    ) -> InferenceResult:
        if self._closed:
            raise ClientClosedError("Cancelled", "client closed")
        try:
            raw = json.dumps({
                "model": model,
                "request": {"messages": list(messages), "schema": dict(schema), "schema_name": schema_name},
                "reasoning_effort": reasoning_effort, "service_tier": service_tier,
                "timeout": timeout, "idle_timeout": idle_timeout,
                "stall_timeout": stall_timeout, "validation_retries": validation_retries,
            }, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError("InvalidRequest", str(exc)) from exc
        try:
            result = json.loads(await self._native.infer(raw))
        except NativeError as exc:
            kind, message = exc.args
            raise _ERRORS.get(kind, InferenceError)(kind, message) from exc
        return InferenceResult(**{**result, "usage": Usage(**result["usage"])})

