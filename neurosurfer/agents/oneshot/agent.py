"""The Agent implementation — single bounded interaction.

Two modes, chosen at construction:

- **Structured** (``output_schema=`` set): one call that returns a validated Pydantic
  model, via :func:`~neurosurfer.agents.runtime.structured.structured_completion`
  (native tool-use under the hood ⇒ valid JSON, with a repair loop).
- **Text / tools** (no schema): one model call; if the model requests tools, run them
  through the permission-gated path and make a synthesis call — bounded by
  ``max_tool_rounds`` (default 1, so at most two model calls total).

For multi-step autonomy use :class:`~neurosurfer.agents.agentic_loop.AgenticLoop`;
for non-function-calling providers use :class:`~neurosurfer.agents.react.ReactAgent`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from neurosurfer.agents.base import BaseAgent
from neurosurfer.agents.conversation import events
from neurosurfer.agents.runtime.loop import execute_tool_uses
from neurosurfer.agents.runtime.structured import structured_completion


class Agent(BaseAgent):
    """A single bounded call (+ optional tools / structured output)."""

    def __init__(
        self,
        *,
        output_schema: type[BaseModel] | None = None,
        max_tool_rounds: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.output_schema = output_schema
        self.max_tool_rounds = max_tool_rounds
        # The result of the last run: a BaseModel (structured) or str (text).
        self.result: Any = None
        #: The round budget ran out while the model was still asking for tools.
        #:
        #: Distinct from finishing, and it used to be indistinguishable. The loop
        #: stops on `rounds >= max_tool_rounds` and takes `response.text()` — but
        #: a turn that is *only* tool calls has no text, so a model cut off
        #: mid-plan returned `""` and the caller reported success. "Fetch the page
        #: then write it to a file" is two rounds and gets one, so the commonest
        #: two-step request produced a green run with a blank answer.
        #:
        #: Set here rather than raised: the agent's job is to report what
        #: happened, and whether a partial answer is acceptable belongs to whoever
        #: asked. `run_base_node` is the one that decides.
        self.cut_short: bool = False

    async def complete(self, user_input: str) -> Any:
        """Run to completion and return the result directly.

        Returns a validated model instance in structured mode, else the final text.
        """
        async for _ in self.run(user_input):
            pass
        return self.result

    async def _run(self, user_input: str) -> AsyncIterator[events.Event]:
        self.history.add_user_text(user_input)

        if self.output_schema is not None:
            model = await structured_completion(
                self.provider,
                self.output_schema,
                user=user_input,
                system=self._effective_system(),
                config=self.gen_config,
            )
            self.result = model
            text = model.model_dump_json()
            yield events.TextDelta(text)
            yield events.RunFinished("completed", text)
            return

        rounds = 0
        while True:
            response = await self.provider.complete(
                self.history.messages,
                self._effective_system(),
                self.tools.schemas(),
                self.gen_config,
            )
            self.usage = self.usage.add(response.usage)
            sent = self.history.snapshot()  # messages the model saw this turn
            self.history.add_assistant_response(response)
            yield events.TurnCompleted(
                response.usage, response.stop_reason,
                input=sent, output=response.as_message(),
            )

            tool_uses = response.tool_uses()
            if not tool_uses or rounds >= self.max_tool_rounds:
                # Two different endings share this branch: the model was done, and
                # the model still wanted tools but has no rounds left. Only the
                # second is a truncation, and telling them apart needs `tool_uses`
                # — which is why the flag is set here rather than inferred later
                # from an empty string.
                self.cut_short = bool(tool_uses)
                text = response.text()
                self.result = text
                if text:
                    yield events.TextDelta(text)
                yield events.RunFinished("cut_short" if self.cut_short else "completed", text)
                return

            rounds += 1
            for tu in tool_uses:
                yield events.ToolStarted(
                    tu.id, tu.name, tu.input, title=self.tools.progress_message(tu.name, tu.input)
                )
            outcomes = await execute_tool_uses(
                tool_uses,
                tools=self.tools,
                ctx=self._ctx,
                permissions=self.permissions,
                mode=self.mode,
            )
            for oc in outcomes:
                yield events.ToolFinished(oc.id, oc.name, oc.result)
            self.history.add_tool_results(
                [(oc.id, oc.result.content, oc.result.is_error) for oc in outcomes]
            )
