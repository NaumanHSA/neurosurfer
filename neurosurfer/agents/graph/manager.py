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
        config: ManagerConfig = None,
        tracer: Optional[Tracer] = None,
        logger: Optional[logging.Logger] = None,
        log_traces: bool = True,
    ):
        self.id = "manager"
        self.llm = llm
        self.config = config or ManagerConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.log_traces = log_traces
        self.tracer = tracer or Tracer(
            config=TracerConfig(enabled=True, log_steps=log_traces)
        )

    def compose_user_prompt(
        self,
        node: GraphNode,
        graph_inputs: Dict[str, Any],
        dependency_results: Dict[str, Any],
        previous_result: Optional[Any],
        *,
        temperature: Optional[float],
        max_new_tokens: Optional[int],
    ) -> str:
        """
        Compose the next `user_prompt` for the given node.

        Returns a plain string to pass into `Agent.run(...)`.
        """
        if self.log_traces:
            rprint(f"\n\\[{self.id}] Tracing Start!")

        purpose = node.purpose or ""
        goal = node.goal or ""
        expected = node.expected_result or ""
        tools = ", ".join(node.tools) if node.tools else "(none)"
        prev_txt = "" if previous_result is None else str(previous_result)
        user_prompt = COMPOSE_NEXT_AGENT_PROMPT_TEMPLATE.format(
                node_id=node.id,
                purpose=purpose,
                goal=goal,
                expected=expected,
                tools=tools,
                graph_inputs=graph_inputs,
                dependency_results=dependency_results,
                prev_txt=prev_txt,
        )
        llm_params = {
            "user_prompt": user_prompt,
            "system_prompt": MANAGER_SYSTEM_PROMPT,
            "temperature": temperature or self.config.temperature,
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "stream": False,
        }
        with self.tracer(
            agent_id=self.id,
            kind="llm.call",
            label="manager.compose_user_prompt",
            inputs={
                "system_prompt_len": len(MANAGER_SYSTEM_PROMPT),
                "user_prompt_len": len(user_prompt),
                **llm_params,
            },
        ) as t:
            next_agent_user_prompt = self.llm.ask(**llm_params).choices[0].message.content.strip()
            # add results from dependency nodes as context to the next_agent_user_prompt
            if dependency_results:
                next_agent_user_prompt = "".join([
                    next_agent_user_prompt,
                    "\n\n# Context from dependency nodes:\n",
                    "\n\n".join([
                        f"## {node_id}\n{result}" for node_id, result in dependency_results.items()
                    ]),
                ])
            t.outputs(output=next_agent_user_prompt)

        if self.log_traces:
            rprint(f"\\[{self.id}] Tracing End!\n")  
        return next_agent_user_prompt
