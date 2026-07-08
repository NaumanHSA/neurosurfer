"""Fake SDK transports for provider tests.

Each provider is constructed normally (no network at construction time) and its
``_client`` is swapped for one of these fakes. A *scripted turn* is the same
high-level shape for both providers — ``(text, [(tool_name, tool_input)])`` — so
the parity suite can drive an identical conversation through both adapters.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from neurosurfer.tools.base import ShellApproval

ScriptedTurn = tuple[str, list[tuple[str, dict[str, Any]]]]


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic fake
# ──────────────────────────────────────────────────────────────────────────────
class _FakeAnthropicStream:
    def __init__(self, events: list[Any], final: Any):
        self._events = events
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for ev in self._events:
            yield ev

    async def get_final_message(self):
        return self._final


def _anthropic_turn(turn: ScriptedTurn) -> _FakeAnthropicStream:
    text, tools = turn
    events: list[Any] = []
    content: list[Any] = []
    if text:
        events.append(
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text=text),
            )
        )
        content.append(SimpleNamespace(type="text", text=text))
    for i, (name, args) in enumerate(tools, start=1):
        tool_id = f"toolu_{i}"
        events.append(
            SimpleNamespace(
                type="content_block_start",
                index=i,
                content_block=SimpleNamespace(type="tool_use", id=tool_id, name=name),
            )
        )
        events.append(
            SimpleNamespace(
                type="content_block_delta",
                index=i,
                delta=SimpleNamespace(type="input_json_delta", partial_json=json.dumps(args)),
            )
        )
        content.append(SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args))
    final = SimpleNamespace(
        content=content,
        stop_reason="tool_use" if tools else "end_turn",
        usage=SimpleNamespace(
            input_tokens=42,
            output_tokens=17,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    return _FakeAnthropicStream(events, final)


class FakeAnthropicMessages:
    def __init__(self, turns: list[ScriptedTurn]):
        self._turns = list(turns)
        self.captured_kwargs: list[dict[str, Any]] = []

    def stream(self, **kwargs):
        self.captured_kwargs.append(kwargs)
        return _anthropic_turn(self._turns.pop(0))

    async def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=123)


class FakeAnthropicClient:
    def __init__(self, turns: list[ScriptedTurn]):
        self.messages = FakeAnthropicMessages(turns)


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI fake
# ──────────────────────────────────────────────────────────────────────────────
class _FakeOpenAIChunks:
    def __init__(self, chunks: list[Any]):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


def _openai_turn_chunks(turn: ScriptedTurn) -> list[Any]:
    text, tools = turn
    chunks: list[Any] = []
    if text:
        chunks.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text, tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )
    for i, (name, args) in enumerate(tools):
        # name + id arrives first, then an args fragment (two chunks per call).
        chunks.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=i,
                                    id=f"call_{i}",
                                    function=SimpleNamespace(name=name, arguments=""),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )
        chunks.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=i,
                                    id=None,
                                    function=SimpleNamespace(
                                        name=None, arguments=json.dumps(args)
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
        )
    finish = "tool_calls" if tools else "stop"
    chunks.append(
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=17),
        )
    )
    return chunks


class FakeOpenAICompletions:
    def __init__(self, turns: list[ScriptedTurn]):
        self._turns = list(turns)
        self.captured_kwargs: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.captured_kwargs.append(kwargs)
        return _FakeOpenAIChunks(_openai_turn_chunks(self._turns.pop(0)))


class FakeOpenAIClient:
    def __init__(self, turns: list[ScriptedTurn]):
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions(turns))


# ──────────────────────────────────────────────────────────────────────────────
# Scripted IO handler (for tool + loop tests)
# ──────────────────────────────────────────────────────────────────────────────
class ScriptedIO:
    """Deterministic IOHandler: queued answers, fixed approvals, captured notices."""

    def __init__(
        self,
        answers: list[str] | None = None,
        approve_plan: bool = True,
        approve_shell: bool = True,
        shell_feedback: str | None = None,
        write_choice: str = "deny",
    ):
        self.answers = list(answers or [])
        self.approve_plan = approve_plan
        self.approve_shell = approve_shell
        # Free-text redirect returned on a denied shell/network/MCP gate.
        self.shell_feedback = shell_feedback
        # Out-of-scope writes default to "deny" in tests (preserving the
        # "guardrail blocks out-of-scope write" semantics); opt in per-test with
        # write_choice="once" or "always".
        self.write_choice = write_choice
        self.notices: list[str] = []
        self.asked: list[str] = []
        self.shell_requests: list[str] = []
        self.write_requests: list[str] = []

    async def ask(self, question: str, options=None) -> str:
        self.asked.append(question)
        return self.answers.pop(0) if self.answers else "ok"

    async def request_plan_approval(self, plan: str) -> tuple[bool, str]:
        return (self.approve_plan, "")

    async def request_shell_approval(self, command: str, reason: str) -> ShellApproval:
        self.shell_requests.append(command)
        feedback = None if self.approve_shell else self.shell_feedback
        return ShellApproval(self.approve_shell, feedback)

    async def request_write_approval(self, path: str, summary: str) -> str:
        self.write_requests.append(path)
        return self.write_choice

    def notify(self, message: str) -> None:
        self.notices.append(message)


# ──────────────────────────────────────────────────────────────────────────────
# Scripted Provider (drives the agent loop without any SDK)
# ──────────────────────────────────────────────────────────────────────────────
class ScriptedProvider:
    """A Provider that replays scripted canonical turns. Each call to ``stream``
    pops the next turn; tool-call ids are generated deterministically."""

    def __init__(self, turns: list[ScriptedTurn], context_window: int = 200_000):
        from neurosurfer.llm.capabilities import ProviderCapabilities

        self._turns = list(turns)
        self.model = "scripted"
        self.calls = 0
        self.capabilities = ProviderCapabilities(
            supports_thinking=False,
            supports_prompt_cache=False,
            supports_token_count=True,
            tool_call_style="anthropic",
            context_window=context_window,
            max_output_tokens=4096,
        )

    async def stream(self, messages, system, tools, config):
        from neurosurfer.llm.types import (
            CanonicalResponse,
            Done,
            TextBlock,
            TextDelta,
            ToolUseArgsDelta,
            ToolUseBlock,
            ToolUseStart,
            Usage,
        )

        self.calls += 1
        text, tool_calls = self._turns.pop(0) if self._turns else ("", [])
        content = []
        if text:
            yield TextDelta(text=text)
            content.append(TextBlock(text=text))
        for i, (name, args) in enumerate(tool_calls):
            import json

            tid = f"call_{self.calls}_{i}"
            yield ToolUseStart(index=i, id=tid, name=name)
            yield ToolUseArgsDelta(index=i, partial_json=json.dumps(args))
            content.append(ToolUseBlock(id=tid, name=name, input=args))
        stop = "tool_use" if tool_calls else "end_turn"
        yield Done(
            response=CanonicalResponse(
                content=content,
                stop_reason=stop,
                usage=Usage(input_tokens=10, output_tokens=5),
                model=self.model,
            )
        )

    async def complete(self, messages, system, tools, config):
        from neurosurfer.llm.types import Done

        response = None
        async for ev in self.stream(messages, system, tools, config):
            if isinstance(ev, Done):
                response = ev.response
        return response

    async def count_tokens(self, messages, system, tools):
        from neurosurfer.llm.tokens import estimate_messages_tokens

        return estimate_messages_tokens(messages, system, tools)
