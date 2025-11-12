# neurosurfer/agents/graph/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

# Existing types you already had:
# - Graph, Node, Ref, GraphResult, GraphSpec, NodeSpec
# We extend Node/NodeSpec with `agent` and `tools` (non-breaking; defaults safe).

@dataclass
class AgentDecl:
    purpose: str = ""     # what this node tries to do
    goal: str = ""        # concrete objective
    expected: str = ""    # short description of expected result

@dataclass
class Policy:
    retries: int = 1
    timeout_s: int = 60
    budget: Dict[str, Any] = field(default_factory=lambda: {"max_new_tokens": 512, "temperature": 0.2})
    model_hint: Optional[str] = None
    backoff: str = "exp"          # "fixed" or "exp"
    backoff_base: float = 0.2     # seconds

@dataclass
class Node:
    id: str
    kind: str = "task"                        # "task" | "map"
    fn: Any = ""                              # str tool id OR "llm.call" OR callable
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)    # ["text"] or ["name: type", ...] or dict (back-compat)
    map_over: Optional[str] = None            # for kind="map"
    policy: Policy = field(default_factory=Policy)

    # NEW (optional)
    agent: AgentDecl = field(default_factory=AgentDecl)
    tools: List[str] = field(default_factory=list)      # presence => router mode for llm.call

@dataclass
class Graph:
    nodes: List[Node]
    outputs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphResult:
    ok: bool
    context: Dict[str, Any]
    outputs: Dict[str, Any]
    errors: Dict[str, str]
