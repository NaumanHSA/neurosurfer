from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import dataclass

from neurosurfer.models.chat_models.base import BaseModel as ChatBaseModel
from .schema import GraphNode
from .templates import MANAGER_SYSTEM_PROMPT

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

    def __init__(self, llm: ChatBaseModel, config: ManagerConfig = None):
        self.llm = llm
        self.config = config or ManagerConfig()

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
        purpose = node.purpose or ""
        goal = node.goal or ""
        expected = node.expected_result or ""
        tools = ", ".join(node.tools) if node.tools else "(none)"
        prev_txt = "" if previous_result is None else str(previous_result)
        user = (
            "You are preparing a prompt for the next agent in a workflow.\n\n"
            f"NODE_ID: {node.id}\n"
            f"NODE_PURPOSE: {purpose}\n"
            f"NODE_GOAL: {goal}\n"
            f"NODE_EXPECTED_RESULT: {expected}\n"
            f"NODE_TOOLS: {tools}\n\n"
            "GRAPH_INPUTS (as JSON-ish):\n"
            f"{graph_inputs}\n\n"
            "DEPENDENCY_RESULTS (node_id -> result):\n"
            f"{dependency_results}\n\n"
            "PREVIOUS_RESULT (may be empty if none):\n"
            f"{prev_txt}\n\n"
            "Compose the next user_prompt string that this node's agent should receive.\n"
            "Return ONLY that prompt text."
        )

        resp = self.llm.ask(
            user_prompt=user,
            system_prompt=MANAGER_SYSTEM_PROMPT,
            temperature=temperature or self.config.temperature,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
