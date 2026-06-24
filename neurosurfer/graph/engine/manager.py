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
        user_intent = str(graph_inputs.get("user_intent", "(not specified)"))

        # Other graph inputs (clarifying answers, etc.) listed separately.
        # Skip internal plumbing keys that are surfaced elsewhere (e.g.
        # `available_tools` is interpolated into the system prompt already).
        _internal = {"user_intent", "available_tools"}
        extra_lines = [
            f"  {k}: {v}"
            for k, v in graph_inputs.items()
            if k not in _internal
        ]

        # Only include deps this node declared.
        depends_on = getattr(node, "depends_on", None) or []
        if depends_on:
            dependency_results = {k: v for k, v in dependency_results.items() if k in depends_on}

        mode = node.mode.value if hasattr(node.mode, "value") else str(node.mode)

        parts: list[str] = [f"User request: {user_intent}"]

        if extra_lines:
            parts.append("Additional inputs:\n" + "\n".join(extra_lines))

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
