from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .schema import GraphNode


@dataclass
class ManagerConfig:
    pass  # reserved for future options


class ManagerAgent:
    """Builds the user_prompt for each node directly — no intermediate LLM.

    An earlier version called a manager LLM to "translate" node specs + dep
    context into agent instructions. That translation was the root cause of
    context drift ("document code repos" → "top code repo tools"). Replaced
    with deterministic prompt assembly that always leads with the verbatim
    user request and appends dependency outputs unchanged.
    """

    def __init__(
        self,
        llm: Any = None,  # kept for API compat; ignored
        id: str = "manager",
        config: ManagerConfig | None = None,
        tracer: Any = None,
        logger: logging.Logger | None = None,
        log_traces: bool = True,
    ) -> None:
        self.id = id
        self.config = config or ManagerConfig()
        self.logger = logger or logging.getLogger(__name__)

    def compose_user_prompt(
        self,
        node: GraphNode,
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
        previous_result: Any,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        # `user_intent` is the Architect's own graph's key, and for a long time it
        # was the *only* one this method looked for — so every workflow that was
        # not the Architect opened its prompt with "User request: (not
        # specified)". Optional now: a graph that declares it gets the header, a
        # graph that does not is simply given its inputs.
        user_intent = graph_inputs.get("user_intent")

        # Skip internal plumbing surfaced elsewhere (`available_tools` is already
        # interpolated into the system prompt).
        _internal = {"user_intent", "available_tools"}

        # Only include deps this node declared.
        depends_on = getattr(node, "depends_on", None) or []
        if depends_on:
            dependency_results = {k: v for k, v in dependency_results.items() if k in depends_on}

        # **An input node's output is a graph input.** `_run_input_node` writes the
        # value it collected under its own id, so the same string arrived here
        # twice — once as an input and once as "context from a previous node" —
        # and was printed twice under headings that disagreed about what it was.
        # Shown once, as the input it is, because that is the name the author
        # gave it and the name any placeholder refers to.
        echoed = {
            dep_id
            for dep_id, value in dependency_results.items()
            if any(value is v or value == v for k, v in graph_inputs.items() if k not in _internal)
        }
        dependency_results = {
            k: v for k, v in dependency_results.items() if k not in echoed
        }

        input_lines = [
            f"  {k}: {v}" for k, v in graph_inputs.items() if k not in _internal
        ]

        mode = node.mode.value if hasattr(node.mode, "value") else str(node.mode)

        parts: list[str] = []
        if user_intent:
            parts.append(f"User request: {user_intent}")

        if input_lines:
            # "Additional" only when there is something for it to be additional
            # *to*. On a hand-built workflow these are the whole request.
            heading = "Additional inputs:" if user_intent else "Inputs:"
            parts.append(heading + "\n" + "\n".join(input_lines))

        if mode == "structured":
            parts.append(
                "Output contract: return STRICT JSON only, matching the schema in "
                "the system prompt. No markdown fences, no explanation, no extra text."
            )

        dep_block = self._format_dependency_context(dependency_results)
        if dep_block:
            parts.append(dep_block)

        return "\n\n".join(parts)

    def _format_dependency_context(self, dependency_results: dict[str, Any]) -> str:
        if not dependency_results:
            return ""

        blocks: list[str] = ["Context from previous nodes:"]
        for node_id, result in dependency_results.items():
            blocks.append(f"--- {node_id} ---")
            blocks.append(str(result))
        return "\n".join(blocks)
