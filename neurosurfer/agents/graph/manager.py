from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import dataclass
import logging

from neurosurfer.models.chat_models.base import BaseChatModel as BaseChatModel
from neurosurfer.tracing import Tracer, TracerConfig
from .schema import GraphNode
from .templates import MANAGER_SYSTEM_PROMPT, COMPOSE_NEXT_AGENT_PROMPT_TEMPLATE
from ..common.utils import rprint

@dataclass
class ManagerConfig():
    temperature: float = 0.7
    max_new_tokens: int = 4096

class ManagerAgent:
    """
    LLM-based manager that composes the `user_prompt` for each node's Agent.

    It does NOT execute tools or call the underlying Agents itself;
    it only crafts prompts based on:
      - Node spec (purpose / goal / expected_result / tools)
      - Original graph inputs
      - Dependency results
      - Previous node result
    """

    def __init__(
        self, 
        llm: BaseChatModel, 
        id: str = "manager",
        config: ManagerConfig = None,
        tracer: Optional[Tracer] = None,
        logger: Optional[logging.Logger] = None,
        log_traces: bool = True,
    ):
        self.id = id
        self.llm = llm
        self.config = config or ManagerConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.log_traces = log_traces
        self.tracer = tracer or Tracer(
            config=TracerConfig(enabled=True, log_steps=log_traces)
        )

    def compose_user_prompt(
        self,
        node: "GraphNode",
        graph_inputs: Dict[str, Any],
        dependency_results: Dict[str, Any],
        previous_result: Optional[Any],  # unused; safe to remove later
        *,
        temperature: Optional[float],
        max_new_tokens: Optional[int],
    ) -> str:
        purpose = node.purpose or ""
        goal = node.goal or ""
        expected = node.expected_result or ""
        tools = ", ".join(node.tools) if node.tools else "(none)"
        mode = node.mode.value

        # OPTIONAL but strongly recommended:
        # Only pass deps that this node depends_on (prevents irrelevant huge context)
        depends_on = getattr(node, "depends_on", None) or []
        if depends_on:
            dependency_results = {k: v for k, v in dependency_results.items() if k in depends_on}

        max_content_chars = 1000
        dependency_node_results = "" if dependency_results else "(none)"
        for node_id, result in dependency_results.items():
            content = f"{result[:1000]}..." if len(result) > max_content_chars else result
            dependency_node_results += f"**NODE: {node_id}**: {content}\n"

        manager_prompt = COMPOSE_NEXT_AGENT_PROMPT_TEMPLATE.format(
            purpose=purpose,
            goal=goal,
            expected=expected,
            mode=mode,
            tools=tools,
            graph_inputs=str(graph_inputs),
            dependency_node_results=dependency_node_results,
        )
        with self.tracer(
            agent_id=self.id,
            kind="llm.call",
            label="manager.compose_user_prompt",
            start_message=f"\nPreparing Next Node...",
            end_message=f"Next Node Ready For Execution!",
            inputs={
                "system_prompt_len": len(MANAGER_SYSTEM_PROMPT),
                "user_prompt_len": len(manager_prompt),
                "mode": mode,
                "dependency_results": dependency_node_results,
            },
        ) as t:
            llm_params = {
                "user_prompt": manager_prompt,
                "system_prompt": MANAGER_SYSTEM_PROMPT,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "max_new_tokens": max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens,
                "stream": False,
            }
            instructions = self.llm.ask(**llm_params).choices[0].message.content.strip()
            dep_context = self._format_dependency_context(dependency_results)

            # Final prompt = instructions + verbatim dependency context
            next_agent_user_prompt = "\n\n".join([instructions, dep_context]).strip()
            t.outputs(output=next_agent_user_prompt)
        return next_agent_user_prompt

    def _format_dependency_context(self, dependency_results: Dict[str, Any]) -> str:
        if not dependency_results:
            return "<BEGIN_DEPENDENCY_CONTEXT>\n(none)\n<END_DEPENDENCY_CONTEXT>"

        blocks = ["<BEGIN_DEPENDENCY_CONTEXT>"]
        for node_id, result in dependency_results.items():
            blocks.append(f"<BEGIN_DEP node_id={node_id}>")
            blocks.append(str(result))
            blocks.append(f"<END_DEP node_id={node_id}>")
            blocks.append("")  # spacing
        blocks.append("<END_DEPENDENCY_CONTEXT>")
        return "\n".join(blocks)