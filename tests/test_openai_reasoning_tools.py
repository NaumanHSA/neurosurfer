"""A model that refuses function tools at its default reasoning effort.

The newest OpenAI reasoning models 400 on chat-completions the moment a request
carries `tools`, unless `reasoning_effort` is explicitly `"none"`:

    Function tools with reasoning_effort are not supported for gpt-5.6-terra in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

We never sent the parameter, so its default applied and every tool-carrying call
failed — which took the Architect from "works on this model" to "does not exist
on this model". Learned at runtime rather than from a model-name list, because a
list is wrong the week after it is written.
"""

from __future__ import annotations

import pytest

from neurosurfer.llm.providers.openai import (
    OpenAICompatProvider,
    _is_tools_need_reasoning_effort,
)


class _Boom(Exception):
    pass


# ── recognising the one error worth retrying ────────────────────────────────────

def test_the_real_message_is_recognised():
    exc = _Boom(
        "Error code: 400 - {'error': {'message': \"Function tools with "
        "reasoning_effort are not supported for gpt-5.6-terra in "
        "/v1/chat/completions. To use function tools, use /v1/responses or set "
        "reasoning_effort to 'none'.\", 'type': 'invalid_request_error', "
        "'param': 'reasoning_effort'}}"
    )
    assert _is_tools_need_reasoning_effort(exc)


@pytest.mark.parametrize("message", [
    # A value we were asked for and got wrong — retrying with 'none' would paper
    # over a real mistake, so both halves of the match are required.
    "Invalid value for 'reasoning_effort': expected one of low, medium, high",
    # Nothing to do with reasoning at all.
    "Rate limit reached for gpt-5.1 in organization org-x",
    "This model does not support tools",
    "context_length_exceeded",
])
def test_unrelated_failures_are_not_retried(message):
    assert not _is_tools_need_reasoning_effort(_Boom(message))


# ── the flag it sets ────────────────────────────────────────────────────────────

def test_the_default_sends_nothing():
    """Every model that has ever worked here keeps working unchanged: the
    parameter is absent until a 400 asks for it."""
    assert OpenAICompatProvider._reasoning_effort_for_tools == ""


async def test_a_refusal_is_retried_once_and_then_remembered():
    """One call pays for the discovery; every later call carries the parameter
    from the start. Mirrors the opener in `stream`."""
    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider.model = "gpt-5.6-terra"
    provider._reasoning_effort_for_tools = ""

    seen: list[dict] = []

    async def create(**kwargs):
        seen.append(dict(kwargs))
        if "reasoning_effort" not in kwargs:
            raise _Boom(
                "Function tools with reasoning_effort are not supported for "
                "gpt-5.6-terra in /v1/chat/completions."
            )
        return "stream"

    kwargs = {"model": provider.model, "tools": [{"type": "function"}]}

    async def open_stream():
        try:
            return await create(**kwargs)
        except Exception as e:
            if not _is_tools_need_reasoning_effort(e) or "reasoning_effort" in kwargs:
                raise
            provider._reasoning_effort_for_tools = "none"
            kwargs["reasoning_effort"] = "none"
            return await create(**kwargs)

    assert await open_stream() == "stream"
    assert len(seen) == 2, "exactly one retry"
    assert "reasoning_effort" not in seen[0]
    assert seen[1]["reasoning_effort"] == "none"
    assert provider._reasoning_effort_for_tools == "none"
