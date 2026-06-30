"""Lightweight conversational pre-flight for the Workflow Architect.

Runs a multi-turn LLM conversation to collect workflow requirements before
invoking ArchitectBuilder (both live in neurosurfer.architect).  The LLM asks clarifying questions one at a time —
each as a multiple-choice question with exactly 3 options (the CLI adds a
free-text "something else" escape).  When it has enough information it emits a
``build_workflow`` tool call.  No ``input()`` or imperative Q&A inside workflow
nodes.

Usage
-----
::

    convo = ArchitectConversation(provider)
    intent, answers = await convo.run(
        "I want to document my code repos",
        ask=cli_ask,    # async (question, choices) -> answer
        say=cli_say,    # (text) -> None  — narration before a question
    )
    # → ArchitectBuilder(provider).run(intent, answers=answers)
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import (
    CanonicalResponse,
    GenerationConfig,
    Message,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)

AskFn = Callable[[str, list[str]], Awaitable[str]]
SayFn = Callable[[str], None]

_SYSTEM_PROMPT = """\
You are the neurosurfer Architect — a focused assistant whose only job is helping
users design automation workflows.

When a user describes what they want to automate, gather the requirements you need
by asking clarifying questions, then design the workflow.

How to ask questions:
- Ask ONE question at a time using the ask_clarifying_question tool.
- Each question MUST offer exactly 3 concrete, distinct options (the interface adds
  a "something else" free-text escape automatically — do not add your own).
- Cover the dimensions that matter: scope/source, depth of analysis, output format,
  destination, and trigger. Skip a dimension if the user already made it clear.
- Ask at least 2 and at most 5 questions total. Do not over-interrogate.

When you have enough information:
- Call the build_workflow tool with a detailed, actionable intent that expands on
  the user's request using every answer you collected — be specific about scope,
  inputs, outputs, and integration points — plus the answers map.

If the user asks something unrelated to workflow building, politely redirect.
""".strip()

_ASK_QUESTION_TOOL = ToolSchema(
    name="ask_clarifying_question",
    description=(
        "Ask the user ONE multiple-choice clarifying question with exactly 3 options. "
        "Call this repeatedly to gather requirements before building."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Short snake_case identifier for this question (e.g. 'output_format').",
            },
            "question": {
                "type": "string",
                "description": "The question text shown to the user.",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
                "description": "Exactly 3 concrete, distinct options.",
            },
        },
        "required": ["id", "question", "choices"],
    },
)

_BUILD_WORKFLOW_TOOL = ToolSchema(
    name="build_workflow",
    description=(
        "Call this when you have gathered enough requirements to design the workflow. "
        "Provide a refined, detailed intent and the collected answers."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": (
                    "Detailed description of the workflow to build. Expand on the user's "
                    "original request using the answers you collected — be specific about "
                    "scope, inputs, outputs, and integration points."
                ),
            },
            "answers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": (
                    "All clarifying answers collected, keyed by the question IDs you used "
                    "(e.g. {'output_format': 'markdown', 'scope': 'all_repos'})."
                ),
            },
        },
        "required": ["intent", "answers"],
    },
)

_CONFIG = GenerationConfig(
    max_tokens=1024,
    temperature=0.7,
    enable_thinking=False,
    stream=False,
)

# Hard ceiling so a misbehaving model can't loop forever asking questions.
_MAX_TURNS = 8


class ArchitectConversation:
    """Multi-turn conversational pre-flight for ArchitectBuilder.

    :meth:`run` drives the whole conversation, calling ``ask`` for each
    clarifying question and returning ``(intent, answers)`` once the LLM is
    ready to build.
    """

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self._history: list[Message] = []
        self._answers: dict[str, str] = {}

    async def run(
        self,
        initial_input: str,
        *,
        ask: AskFn,
        say: SayFn | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Collect requirements, then return ``(intent, answers)``.

        Parameters
        ----------
        initial_input:
            The user's first message describing what they want to build.
        ask:
            Async callback ``(question, choices) -> answer``.  The CLI renders
            an arrow-key menu of *choices* plus a free-text escape and returns
            the chosen / typed answer.
        say:
            Optional callback to surface the assistant's narration text (the
            sentence it writes alongside a tool call).
        """
        self._history.append(Message.user_text(initial_input))

        for _ in range(_MAX_TURNS):
            response: CanonicalResponse = await self.provider.complete(
                messages=self._history,
                system=_SYSTEM_PROMPT,
                tools=[_ASK_QUESTION_TOOL, _BUILD_WORKFLOW_TOOL],
                config=_CONFIG,
            )
            self._history.append(response.as_message())

            # Surface any narration the model wrote alongside its tool call.
            text = response.text().strip()
            if text and say is not None:
                say(text)

            tool_uses = response.tool_uses()
            if not tool_uses:
                # No tool call — treat the text as a question and fall back to
                # free-text input so the conversation can still progress.
                answer = await ask(text or "Tell me more about what you want:", [])
                self._history.append(self._tool_or_user_reply(None, answer))
                continue

            tool: ToolUseBlock = tool_uses[0]

            if tool.name == "build_workflow":
                intent = str(tool.input.get("intent", "")).strip()
                collected = dict(tool.input.get("answers", {}))
                # Merge with answers we tracked locally (defensive).
                merged = {**self._answers, **collected}
                self._ack_tool(tool.id, "building now")
                return intent or initial_input, merged

            if tool.name == "ask_clarifying_question":
                qid = str(tool.input.get("id", f"q{len(self._answers) + 1}"))
                question = str(tool.input.get("question", ""))
                choices = [str(c) for c in tool.input.get("choices", [])][:3]
                answer = await ask(question, choices)
                self._answers[qid] = answer
                self._history.append(
                    Message(
                        role="user",
                        content=[ToolResultBlock(
                            tool_use_id=tool.id,
                            content=answer or "(no answer)",
                        )],
                    )
                )
                continue

            # Unknown tool — acknowledge and continue.
            self._ack_tool(tool.id, "ok")

        # Hit the turn ceiling without an explicit build — build with what we have.
        return initial_input, dict(self._answers)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _ack_tool(self, tool_use_id: str, content: str) -> None:
        self._history.append(
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id=tool_use_id, content=content)],
            )
        )

    def _tool_or_user_reply(self, tool_use_id: str | None, answer: str) -> Message:
        if tool_use_id is not None:
            return Message(
                role="user",
                content=[ToolResultBlock(tool_use_id=tool_use_id, content=answer or "(no answer)")],
            )
        return Message.user_text(answer or "(no answer)")
