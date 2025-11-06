# neurosurfer/agents/graph/selector.py
from __future__ import annotations
from typing import Dict, Optional, Callable, Any
from .types import Graph

class FlowRegistry:
    def __init__(self):
        self._by_name: Dict[str, Graph] = {}
    def register(self, flow: Graph, *, version: Optional[str] = None):
        key = f"{flow.name}@{version}" if version else flow.name
        self._by_name[key] = flow
    def get(self, name_or_versioned: str) -> Graph:
        return self._by_name[name_or_versioned]

class FlowSelector:
    """
    Simple rule-based selector (extend with semantic routing if you like).
    rules: list of (predicate, route_to) where predicate(query:str)->bool
    """
    def __init__(self, registry: FlowRegistry):
        self.registry = registry
        self.rules: list[tuple[Callable[[str], bool], str]] = []

    def add_rule(self, predicate: Callable[[str], bool], route_to: str):
        self.rules.append((predicate, route_to))

    def pick(self, query: str, default_flow: Optional[str] = None) -> Graph:
        for pred, name in self.rules:
            if pred(query):
                return self.registry.get(name)
        if default_flow:
            return self.registry.get(default_flow)
        raise KeyError("No matching flow and no default provided.")
