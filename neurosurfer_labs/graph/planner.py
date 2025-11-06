# neurosurfer/agents/graph/planner.py
from __future__ import annotations
from typing import Any, Dict
from .errors import PlanningError
from .loader import FlowLoader
from .types import Graph

class PlannerAgent:
    """
    Minimal stub: in real life, call LLM to synthesize YAML from a tool catalog.
    """
    def __init__(self, llm: Any):
        self.llm = llm

    def plan_from_query(self, query: str, *, skeleton: str | None = None) -> Graph:
        if skeleton:
            return FlowLoader.from_yaml(skeleton)
        # As a safe default, return a trivial 1-node flow that echoes the query.
        import textwrap, tempfile, pathlib
        yml = textwrap.dedent(f"""
        name: echo_plan
        nodes:
          - id: echo
            kind: task
            fn: "llm.echo"
            inputs: {{ text: "${{inputs.query}}" }}
            outputs: ["text"]
        outputs: {{ result: "${{echo.text}}" }}
        """).strip()
        tmp = pathlib.Path(tempfile.gettempdir()) / "echo_plan.yml"
        tmp.write_text(yml)
        return FlowLoader.from_yaml(tmp)
