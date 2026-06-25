"""Conversation history with the helpers compaction and the loop rely on.

Kept deliberately thin: the engine mutates a list of canonical ``Message`` objects.
The one non-trivial operation is :meth:`replace_prefix_with_summary`, used by the
context manager to swap an old prefix for a single summary message.
"""

from __future__ import annotations

from neurosurfer.llm.types import (
    CanonicalResponse,
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
)


class MessageHistory:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def snapshot(self) -> list[Message]:
        return list(self._messages)

    def add_user_text(self, text: str) -> None:
        self._messages.append(Message.user_text(text))

    def add_user_message(self, message: Message) -> None:
        self._messages.append(message)

    def add_assistant_response(self, response: CanonicalResponse) -> None:
        self._messages.append(response.as_message())

    def add_tool_results(
        self,
        results: list[tuple[str, str, bool]],
        images: list[ImageBlock] | None = None,
    ) -> None:
        """Append a user message of tool_result blocks: (tool_use_id, content, is_error).

        Any ``images`` produced by the tools are appended **after** the tool_result
        blocks in the same user turn (Anthropic requires tool_result blocks first, and
        a single turn keeps the user/assistant alternation intact for both providers).
        """
        blocks: list = [
            ToolResultBlock(tool_use_id=tid, content=content, is_error=is_error)
            for (tid, content, is_error) in results
        ]
        if images:
            blocks.extend(images)
        self._messages.append(Message(role="user", content=blocks))

    def add_user_images(self, text: str, images: list[ImageBlock]) -> None:
        """Append a user turn with text + images (direct, non-tool image input)."""
        self._messages.append(Message.user_with_images(text, images))

    def replace_prefix_with_summary(self, keep_tail: int, summary: str) -> None:
        """Replace everything except the last ``keep_tail`` messages with a single
        user message carrying the compaction summary."""
        tail = self._messages[-keep_tail:] if keep_tail > 0 else []
        summary_msg = Message(
            role="user", content=[TextBlock(text=summary)]
        )
        self._messages = [summary_msg, *tail]

    def replace_all(self, messages: list[Message]) -> None:
        self._messages = list(messages)
