# neurosurfer/agents/graph/loader.py
from __future__ import annotations
from typing import Any, Dict, List
from dataclasses import dataclass
from pathlib import Path
import yaml, re

from .types import Graph, Node, NodePolicy, GraphConfig
from .types import Ref
from .errors import ValidationError

_REF_RE = re.compile(r"^\$\{([^}]+)\}$")

def _parse_value(v: Any):
    if isinstance(v, str):
        m = _REF_RE.match(v.strip())
        if m:
            return Ref(m.group(1))
    return v

def _parse_io(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _parse_value(v) for k, v in (obj or {}).items()}

class FlowLoader:
    @staticmethod
    def from_yaml(path: str | Path) -> Graph:
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValidationError("YAML root must be a mapping.")
        name = data.get("name") or Path(path).stem
        cfg = data.get("config", {}) or {}
        graph_cfg = GraphConfig(max_concurrency=int(cfg.get("max_concurrency", 4)))

        nodes = []
        for raw in data.get("nodes", []):
            if not isinstance(raw, dict):
                raise ValidationError("Each node must be a mapping.")
            nid = raw["id"]
            kind = raw.get("kind", "task")
            fn = raw.get("fn", "")
            inputs = _parse_io(raw.get("inputs", {}))
            outputs = _parse_io(raw.get("outputs", {}))
            policy = NodePolicy(
                retries=int(raw.get("policy", {}).get("retries", 1)),
                timeout_s=int(raw.get("policy", {}).get("timeout_s", 60)),
                backoff=str(raw.get("policy", {}).get("backoff", "exponential")),
                backoff_base=float(raw.get("policy", {}).get("backoff_base", 0.6)),
                budget=raw.get("policy", {}).get("budget", {}) or {},
                model_hint=raw.get("policy", {}).get("model_hint"),
                concurrency_group=raw.get("policy", {}).get("concurrency_group"),
            )
            map_over = raw.get("map_over")
            if isinstance(map_over, str) and map_over.startswith("${") and map_over.endswith("}"):
                map_over = map_over[2:-1]
            nodes.append(Node(id=nid, kind=kind, fn=fn, inputs=inputs, outputs=outputs, map_over=map_over, policy=policy))

        outputs = data.get("outputs", {}) or {}
        inputs_schema = data.get("inputs", {}) or {}
        g = Graph(name=name, nodes=nodes, outputs=outputs, inputs_schema=inputs_schema, config=graph_cfg)
        _validate_acyclic(g)
        return g

def _validate_acyclic(graph: Graph):
    # simple check: ensure no duplicate node ids and non-empty
    ids = set()
    for n in graph.nodes:
        if n.id in ids:
            raise ValidationError(f"Duplicate node id: {n.id}")
        ids.add(n.id)
    # full cycle detection is done in executor when building dependency graph from Refs
