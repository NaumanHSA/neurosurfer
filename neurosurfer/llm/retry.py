"""Unified retryable-error categorization across both SDKs.

429 / 500 / 502 / 503 / 529 / timeouts / connection errors are retryable with
backoff (honoring ``retry-after``); auth / quota / malformed-request are terminal.
Context-overflow is special-cased so the agent loop can react with compaction
instead of retrying blindly.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ..observability.logging import get_logger

log = get_logger("llm.retry")

T = TypeVar("T")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_CONTEXT_OVERFLOW_MARKERS = (
    "prompt is too long",
    "prompt too long",
    "context length",
    "context_length_exceeded",
    "maximum context length",
    "too many tokens",
    "exceeds the maximum",
    "input is too long",
)


def _status_code(err: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        val = getattr(err, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(err, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def is_context_overflow_error(err: BaseException) -> bool:
    """True when the error is an input-too-large rejection (reactive-compaction
    trigger), regardless of provider."""
    msg = str(getattr(err, "message", "") or err).lower()
    if any(marker in msg for marker in _CONTEXT_OVERFLOW_MARKERS):
        return True
    # Anthropic surfaces this as a 400 with a specific type.
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        etype = str(body.get("error", {}).get("type", "")).lower()
        if etype in ("invalid_request_error",) and "long" in str(body).lower():
            return True
    return False


def is_retryable_error(err: BaseException) -> bool:
    if isinstance(err, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    name = type(err).__name__.lower()
    if any(s in name for s in ("timeout", "connection", "apiconnection", "internalserver")):
        return True
    if is_context_overflow_error(err):
        return False  # handled by reactive compaction, not retry
    sc = _status_code(err)
    if sc is not None:
        return sc in RETRYABLE_STATUS
    return False


def retry_after_seconds(err: BaseException) -> float | None:
    headers = None
    resp = getattr(err, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None)
    if headers is None:
        headers = getattr(err, "headers", None)
    if not headers:
        return None
    try:
        val = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Run ``fn`` with exponential backoff on retryable errors."""
    attempt = 0
    while True:
        try:
            return await fn()
        except BaseException as err:  # noqa: BLE001
            attempt += 1
            if attempt >= max_attempts or not is_retryable_error(err):
                raise
            delay = retry_after_seconds(err)
            if delay is None:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay += random.uniform(0, delay * 0.25)  # jitter
            log.warning(
                "retryable error (attempt %d/%d): %s — backing off %.1fs",
                attempt,
                max_attempts,
                type(err).__name__,
                delay,
            )
            await asyncio.sleep(delay)
