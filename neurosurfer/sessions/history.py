"""Message history serialization for sessions.

Messages are stored as a single JSON array so the file is always read and
written as a complete unit. Pydantic's discriminated-union round-trip handles
all block types (TextBlock, ThinkingBlock.signature, ToolUseBlock, ToolResultBlock)
without custom serializers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm.types import Message


def save_history(path: Path, messages: list[Message]) -> None:
    """Serialize messages to a JSON array. Atomic write via temp rename."""
    data = [m.model_dump(mode="json") for m in messages]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def load_history(path: Path) -> list[Message]:
    """Deserialize messages. Returns [] on missing or corrupt file — never raises."""
    if not path.exists():
        return []
    try:
        from ..llm.types import Message

        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return [Message.model_validate(d) for d in raw]
    except Exception:  # noqa: BLE001
        return []


def estimate_tokens(messages: list[Message]) -> int:
    """Cheap token estimate: sum text content lengths // 4 (no tiktoken needed)."""
    total = 0
    for m in messages:
        for block in m.content:
            if hasattr(block, "text"):
                total += len(block.text) // 4
            elif hasattr(block, "thinking"):
                total += len(block.thinking) // 4
            elif hasattr(block, "input"):
                total += len(str(block.input)) // 4
            elif hasattr(block, "content"):
                total += len(str(block.content)) // 4
    return total
