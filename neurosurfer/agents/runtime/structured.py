"""Structured output for the native agent stack, via native tool-use (R4 foundation).

The `core/` agent and providers speak native tool-use. The cleanest way to get a
*validated* structured result (e.g. a workflow `plan` node returning a `WorkflowPlan`)
is to hand the model a single synthetic "submit" tool whose input schema **is** the
target pydantic model — the provider then emits schema-shaped JSON, which we validate.

This is the modern alternative to the legacy text-parse-and-repair ReAct path
(`<__final_answer__>` sentinels, JSON repair loops). It is a standalone helper: it does
not touch the agent loop, and it is what R4's `base`-kind workflow nodes will call.
"""

from __future__ import annotations

import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import GenerationConfig, Message, ToolResultBlock, ToolSchema
from neurosurfer.tools.schema import model_to_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_SUBMIT_TOOL = "submit_result"


class StructuredCompletionError(RuntimeError):
    """Raised when a valid structured result could not be obtained within the budget."""


async def structured_completion(
    provider: Provider,
    schema: type[T],
    *,
    user: str,
    system: str | None = None,
    config: GenerationConfig | None = None,
    max_attempts: int = 3,
) -> T:
    """Return an instance of *schema*, produced by *provider* via a submit tool.

    The model is given one tool (``submit_result``) whose input schema is *schema*
    and instructed to call it once. Invalid calls are fed back for repair, up to
    *max_attempts*. Raises :class:`StructuredCompletionError` on exhaustion.
    """
    submit = ToolSchema(
        name=_SUBMIT_TOOL,
        description=f"Submit the final result as a {schema.__name__}. Call this exactly once.",
        input_schema=model_to_schema(schema),
    )
    cfg = config or GenerationConfig(
        max_tokens=4096, temperature=0.2, enable_thinking=False, stream=False
    )
    sys = (system or "").rstrip()
    instruction = (
        f"You MUST call the `{_SUBMIT_TOOL}` tool exactly once with the complete, "
        f"valid result. Do not answer in prose."
    )
    sys = f"{sys}\n\n{instruction}" if sys else instruction

    messages: list[Message] = [Message.user_text(user)]
    last_error: str = ""

    for attempt in range(1, max_attempts + 1):
        response = await provider.complete(messages, sys, [submit], cfg)
        tool_uses = [t for t in response.tool_uses() if t.name == _SUBMIT_TOOL]
        if not tool_uses:
            tool_uses = response.tool_uses()  # accept a misnamed call as a fallback

        if not tool_uses:
            last_error = "model did not call the submit tool"
            messages.append(response.as_message())
            messages.append(
                Message.user_text(f"You did not call `{_SUBMIT_TOOL}`. Call it now with the result.")
            )
            continue

        tu = tool_uses[0]
        try:
            return schema.model_validate(tu.input)
        except ValidationError as exc:
            last_error = _short(str(exc))
            logger.info("structured_completion attempt %d invalid: %s", attempt, last_error)
            messages.append(response.as_message())
            messages.append(
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id=tu.id,
                            content=f"Invalid {schema.__name__}: {last_error}. "
                            f"Fix the fields and call `{_SUBMIT_TOOL}` again.",
                            is_error=True,
                        )
                    ],
                )
            )

    raise StructuredCompletionError(
        f"Could not obtain a valid {schema.__name__} after {max_attempts} attempts: {last_error}"
    )


def _short(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "…"
