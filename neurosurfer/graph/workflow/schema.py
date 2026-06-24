"""Public schema types for the workflow layer.

Re-exports the vendored runtime types that external code should import from here
(not from the private ``_runtime`` path), and adds the neurosurfer-level
``WorkflowManifest`` that describes the multi-file package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neurosurfer.graph.engine import (  # noqa: F401
    _VALID_NODE_KINDS,
    Graph,
    GraphExecutionResult,
    GraphInput,
    GraphNode,
    NodeExecutionResult,
    NodeMode,
)

__all__ = [
    # vendored re-exports
    "Graph",
    "GraphExecutionResult",
    "GraphInput",
    "GraphNode",
    "NodeExecutionResult",
    "NodeMode",
    "_VALID_NODE_KINDS",
    # manifest
    "WorkflowManifest",
]


@dataclass
class WorkflowManifest:
    """Top-level manifest stored in ``workflow.yaml`` for a multi-file package."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    entrypoint: str = "graph.yaml"
    created_by: str | None = None
    created_at: str | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> WorkflowManifest:
        return cls(
            name=d["name"],
            version=str(d.get("version", "1.0.0")),
            description=d.get("description", ""),
            entrypoint=d.get("entrypoint", "graph.yaml"),
            created_by=d.get("created_by"),
            created_at=d.get("created_at"),
            tags=list(d.get("tags") or []),
        )

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entrypoint": self.entrypoint,
        }
        if self.created_by:
            d["created_by"] = self.created_by
        if self.created_at:
            d["created_at"] = self.created_at
        if self.tags:
            d["tags"] = self.tags
        return d
