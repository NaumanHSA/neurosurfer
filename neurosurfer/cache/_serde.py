"""Serialization helpers for CanonicalResponse (used by DiskResponseCache)."""
from __future__ import annotations

import json
from typing import Annotated, Union

from pydantic import Field, TypeAdapter

from neurosurfer.llm.types import (
    CanonicalResponse,
    ContentBlock,
    Usage,
)

_BLOCK_ADAPTER: TypeAdapter[ContentBlock] = TypeAdapter(
    Annotated[ContentBlock, Field(discriminator="type")]
)


def serialize(response: CanonicalResponse) -> str:
    return json.dumps({
        "content": [b.model_dump() for b in response.content],
        "stop_reason": response.stop_reason,
        "usage": response.usage.model_dump(),
        "model": response.model,
    })


def deserialize(raw: str | dict) -> CanonicalResponse:
    d = json.loads(raw) if isinstance(raw, str) else raw
    blocks = [_BLOCK_ADAPTER.validate_python(b) for b in d["content"]]
    return CanonicalResponse(
        content=blocks,
        stop_reason=d["stop_reason"],
        usage=Usage(**d["usage"]),
        model=d["model"],
    )
