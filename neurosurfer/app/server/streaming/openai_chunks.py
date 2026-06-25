from __future__ import annotations

from ..schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)


def chunk_role(*, id: str, created: int, model: str, role: str = "assistant") -> dict:
    return ChatCompletionChunk(
        id=id,
        created=created,
        model=model,
        choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(role=role))],
    ).model_dump()


def chunk_text(*, id: str, created: int, model: str, text: str) -> dict:
    return ChatCompletionChunk(
        id=id,
        created=created,
        model=model,
        choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(content=text))],
    ).model_dump()


def chunk_end(*, id: str, created: int, model: str, finish_reason: str = "stop") -> dict:
    return ChatCompletionChunk(
        id=id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason=finish_reason,
            )
        ],
    ).model_dump()
