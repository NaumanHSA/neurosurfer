# neurosurfer/agents/graph/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union

OutputsSpec = Union[
    List[str],            # e.g. ["text"] -> free text
    Dict[str, Any],       # e.g. {"num1": "float", "meta": {"$object": {...}}, "tags": {"$array": "str"}}
]

class Ref:
    """Reference to another node's output, or inputs.*"""
    def __init__(self, path: str):
        self.path = path  # e.g., "plan.subtopics", "inputs.topic"
    def __repr__(self) -> str:
        return f"Ref({self.path!r})"

NodeKind = Literal["task", "map", "join"]

@dataclass
class NodePolicy:
    retries: int = 1
    timeout_s: int = 60
    backoff: str = "exponential"  # or "fixed"
    backoff_base: float = 0.6
    budget: Dict[str, Any] = field(default_factory=dict)  # e.g., {"max_new_tokens": 512, "temperature": 0.2}
    model_hint: Optional[str] = None
    concurrency_group: Optional[str] = None   # for shared throttling groups

@dataclass
class Node:
    id: str
    kind: NodeKind = "task"
    fn: str | Any = ""                 # string name resolved via registry/toolkit OR callable
    inputs: Dict[str, Any] = field(default_factory=dict)    # literals and/or Ref(...)
    outputs: Dict[str, Any] = field(default_factory=dict)        # keys produced by this node
    map_over: Optional[str] = None     # for kind="map": Ref path to list; ex: "plan.subtopics"
    policy: NodePolicy = field(default_factory=NodePolicy)

@dataclass
class NodeSpec:
    id: str
    kind: Literal["task", "join"]
    fn: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: OutputsSpec = field(default_factory=list)
    policy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphConfig:
    max_concurrency: int = 4

@dataclass
class Graph:
    name: str
    nodes: List[Node]
    inputs_schema: Dict[str, Any] = field(default_factory=dict)  # optional, for validation/docs
    outputs: Dict[str, Any] = field(default_factory=dict)        # name -> Ref path
    config: GraphConfig = field(default_factory=GraphConfig)

@dataclass
class GraphSpec:
    name: str
    inputs: Dict[str, str] = field(default_factory=dict)
    nodes: List[NodeSpec] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphResult:
    ok: bool
    context: Dict[str, Any]
    outputs: Dict[str, Any]
    errors: Dict[str, str] = field(default_factory=dict)
