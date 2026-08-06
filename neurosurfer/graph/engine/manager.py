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
        task: str,
        dependency_results: dict[str, Any],
        previous_result: Any = None,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """The turn a node is given: what to do, then what fed into it.

        ## What is deliberately not here any more

        Every graph input, printed for every node, under an `Inputs:` heading.
        It went because it could not be made correct — only less wrong:

        - a `map` body was handed the whole collection it was iterating over,
          once per item, beside the single item it was working on;
        - a value the instruction had already interpolated was then printed
          again underneath it, which is Phase 6's defect in a third place;
        - and none of it was *asked for*. A node received every input because
          the graph had them, not because the step needed them.

        Narrowing it kept running into the same wall: any rule for "which
        inputs matter to this node" is a worse version of a rule the author has
        already written, in the placeholders of the instruction itself.

        So the contract is now the small one, and it matches what a LangGraph
        node gets: **what the task text names, plus the outputs of the steps it
        declared as dependencies.** Nothing ambient. A node that names nothing
        and depends on nothing is a node with no input, which is a defect the
        validator reports before the run rather than something the engine papers
        over by reciting the whole graph at it.
        """
        parts: list[str] = [task]

        # Only include deps this node declared. `depends_on` is the whole of
        # what a node inherits — the rest of the graph is not its business.
        depends_on = getattr(node, "depends_on", None) or []
        if depends_on:
            dependency_results = {
                k: v for k, v in dependency_results.items() if k in depends_on
            }

        mode = node.mode.value if hasattr(node.mode, "value") else str(node.mode)
        if mode == "structured":
            parts.append(
                "Output contract: return STRICT JSON only, matching the schema in "
                "the system prompt. No markdown fences, no explanation, no extra text."
            )

        dep_block = self._format_dependency_context(dependency_results)
        if dep_block:
            parts.append(dep_block)

        return "\n\n".join(p for p in parts if p.strip())

    def _format_dependency_context(self, dependency_results: dict[str, Any]) -> str:
        if not dependency_results:
            return ""

        blocks: list[str] = ["Context from previous nodes:"]
        for node_id, result in dependency_results.items():
            blocks.append(f"--- {node_id} ---")
            blocks.append(str(result))
        return "\n".join(blocks)
