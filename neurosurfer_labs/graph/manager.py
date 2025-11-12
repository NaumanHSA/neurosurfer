# neurosurfer_labs/graph/manager.py
from __future__ import annotations
from typing import Any, Dict, Optional
from .templates import MANAGER_SYSTEM
from neurosurfer.models.chat_models import BaseModel


class ManagerAgent:
    def __init__(self, llm: BaseModel):
        self.llm = llm

    def make_prompt(
        self,
        agent_spec: Dict[str, str],
        graph_inputs: Dict[str, Any],
        prev_result: Dict[str, Any] | None,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:
        """
        agent_spec = {"purpose":..., "goal":..., "expected":...}
        prev_result envelope example:
          {
            "selected_tool": "calculator",
            "result": "294",
            "inputs": { ... },
            "returns": None,
            "text": "...",          # (for plain LLM nodes)
            ...
          }
        """
        purpose = agent_spec.get("purpose","")
        goal = agent_spec.get("goal","")
        expected = agent_spec.get("expected","")

        # Build a concise sketch of prior outputs
        prior = ""
        if prev_result:
            # Prefer tool result; then text
            core = prev_result.get("result") or prev_result.get("text") or ""
            try:
                core_str = core if isinstance(core, str) else str(core)
            except Exception:
                core_str = ""
            prior = core_str[:4000]

        user = (
            "NEXT_AGENT_CONTEXT\n"
            f"PURPOSE:\n{purpose}\n\n"
            f"GOAL:\n{goal}\n\n"
            f"EXPECTED_RESULT:\n{expected}\n\n"
            f"USER_INPUTS (as JSON):\n{graph_inputs}\n\n"
            f"PREVIOUS_RESULT (text or JSON):\n{prior}\n\n"
            "Compose the next user_prompt string that the agent should receive."
        )

        resp = self.llm.ask(
            user_prompt=user,
            system_prompt=MANAGER_SYSTEM,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=False
        )
        return (resp.choices[0].message.content or "").strip()
