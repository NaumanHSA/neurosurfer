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

_PREAMBLE = """\
You are the neurosurfer Architect — a focused assistant whose only job is helping
users design automation workflows.
""".strip()

_HOW_TO_ASK = """\
How to ask questions:
- Ask ONE question at a time using the ask_clarifying_question tool.
- Each question MUST offer exactly 3 concrete, distinct options (the interface adds
  a "something else" free-text escape automatically — do not add your own).
- Cover the dimensions that matter: scope/source, depth of analysis, output format,
  destination, and trigger. Skip a dimension if the user already made it clear.
"""

_CLOSING = """\
When you have enough information:
- Call the build_workflow tool with a precise, actionable intent that restates the
  user's request and folds in every answer you collected — specific about scope,
  inputs, outputs and integration points — plus the answers map.

**Add nothing the user did not ask for.** The intent is not a proposal, it is a
specification: every clause becomes a planned step and an acceptance criterion the
build is judged against. One workflow asked only for a markdown file; the intent
added "return the content as the workflow result for visibility", that became a
criterion demanding an object with particular keys, and the build failed against a
requirement nobody had.

So: no extra outputs, no intermediate artefacts, no logging or error-handling
sections, no payload shapes or field names the user never mentioned. Where the
request is silent on something the design needs, write it as an assumption
("assume the report covers all centres") rather than as a requirement.

If the user asks something unrelated to workflow building, politely redirect.
"""

# `always` — the original behaviour, kept for someone who wants to be asked.
_RULES_ALWAYS = """\
When a user describes what they want to automate, gather the requirements you need
by asking clarifying questions, then design the workflow.
""" + _HOW_TO_ASK + """\
- Ask at least 2 and at most 5 questions total. Do not over-interrogate.

""" + _CLOSING

# `auto` — the default. The bar is *the answer would change the workflow*, not
# "more detail would be nice": a specific request interrogated anyway is three
# clicks of friction before anything is built, and the model reliably invents
# questions when told to ask some.
_RULES_AUTO = """\
**Earlier turns may appear above.** If they do, this message is part of that
conversation, not a fresh request. A user who is handed a blocked build and then
pastes a connection string, a path or a key is *answering it* — carry the original
request forward with the new value, and do not ask what they would like to
automate. Only treat a message as a new request when it plainly is one.

When a user describes what they want to automate, decide whether you can already
build it. Building is the default and asking is the exception.

Ask a question ONLY when the request is genuinely ambiguous in a way that changes
what gets built — a fork you cannot pick between, a required destination nobody
named, a format the rest of the design depends on. Do NOT ask to gather
nice-to-have detail, to confirm something already stated, or to offer options
where any choice would do: pick a sensible default and say so in the intent
instead.

NEVER ask **how** a credential is supplied. That is already decided: stored values
are named by the step that needs them and injected into the tool call, and the
user is prompted for any that are unset. Asking invites an answer like "via
environment variables", which becomes a workflow step for reading environment
variables — a capability nothing provides, which blocks the build. *Which*
database or account to use is a fair question; *how the password arrives* is not.

If the request is specific enough to build, call build_workflow IMMEDIATELY with
an empty answers map. Most requests are.

At most 3 questions, and only if each one genuinely changes the design.

""" + _HOW_TO_ASK + "\n" + _CLOSING


def _system_prompt(mode: str) -> str:
    rules = _RULES_ALWAYS if mode == "always" else _RULES_AUTO
    return f"{_PREAMBLE}\n\n{rules}".strip()

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
                    "The workflow to build, stated precisely. Restate the user's "
                    "request and fold in the answers you collected — be specific "
                    "about scope, inputs, outputs and integration points.\n\n"
                    "State ONLY what the user asked for. Do NOT add requirements "
                    "they did not state — no extra outputs 'for visibility', no "
                    "intermediate artefacts, no error-handling or logging "
                    "sections, no field names or payload shapes they never "
                    "mentioned. Everything written here is treated downstream as "
                    "a requirement: it is planned as steps and judged as "
                    "acceptance criteria. A helpful-sounding addition becomes a "
                    "condition the build must satisfy and can fail on.\n\n"
                    "Where the request is genuinely silent on something the "
                    "design needs, say the assumption in one clause — 'assume "
                    "…' — rather than promoting it to a requirement."
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
    enable_thinking=False,
    stream=False,
)

# Hard ceiling so a misbehaving model can't loop forever asking questions.
_MAX_TURNS = 8

#: How many earlier turns to carry. Enough for "it blocked, here is the value it
#: asked for" — the case this exists to serve — without paying for a whole
#: afternoon's transcript on every build.
_HISTORY_TURNS = 8
#: A blocked reason or a refined intent runs to a page. Truncated because what
#: matters is *what happened*, not every clause of it.
_HISTORY_CHARS = 900


def _prior_turns(history: list[dict[str, str]] | None) -> list[Message]:
    """Earlier turns as messages, oldest first.

    Only what a person would consider part of the conversation: what they asked
    for, and what came back. The step log ("node added: …" forty times over) is
    activity, not dialogue — including it would bury the exchange that matters in
    a build's own narration.
    """
    out: list[Message] = []
    for turn in (history or [])[-_HISTORY_TURNS:]:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        if len(text) > _HISTORY_CHARS:
            text = text[:_HISTORY_CHARS] + "…"
        role = str(turn.get("role") or "user").lower()
        out.append(
            Message.user_text(text) if role == "user" else Message.assistant_text(text)
        )
    return out


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
        mode: str = "auto",
        history: list[dict[str, str]] | None = None,
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
        mode:
            ``auto`` (default) asks only where the answer changes the design and
            otherwise builds straight away; ``always`` asks 2-5 questions
            whatever the request. Both may return an empty answers map — in
            ``auto`` that is the expected outcome, and the refined intent is
            still worth the call.
        """
        # Earlier turns of this conversation, if any. Without them every message
        # is a cold start: a build blocked asking for a connection string, the
        # user pasted one, and the Architect replied "what would you like to
        # automate with it?" — because it had never seen the question it was the
        # answer to. The panel is a chat, so the agent has to be given the chat.
        for turn in _prior_turns(history):
            self._history.append(turn)
        self._history.append(Message.user_text(initial_input))

        for _ in range(_MAX_TURNS):
            response: CanonicalResponse = await self.provider.complete(
                messages=self._history,
                system=_system_prompt(mode),
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
