"""Auto-derived neurosurfer self-knowledge for the Architect agent (Phase 3).

:class:`KnowledgeBase` is the single entry point:

- :meth:`manifest` — the full capability manifest (introspected, versioned).
- :meth:`render_context` — a compact markdown block for the agent's system prompt
  (node kinds, expression cheat-sheet, tool one-liners; a few KB, not the world).
- :meth:`search_docs` / :meth:`describe_tool` / :meth:`describe_node_kind` — the
  on-demand retrieval calls that become agent tools in Phase 4, so detail is
  pulled when needed instead of dumped into every prompt.

Everything derives from live code (see ``manifest.py``); the freshness tests in
``tests/test_architect_knowledge.py`` fail if this layer drifts from the engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .docs_index import DocSection, DocsIndex  # noqa: F401
from .manifest import build_manifest, manifest_version  # noqa: F401

__all__ = [
    "KnowledgeBase",
    "DocsIndex",
    "DocSection",
    "build_manifest",
    "manifest_version",
]


class KnowledgeBase:
    """Facade over the capability manifest + docs retrieval."""

    def __init__(self, docs_dir: Path | None = None) -> None:
        self._docs = DocsIndex(docs_dir)
        self._manifest: dict[str, Any] | None = None

    # ── manifest ────────────────────────────────────────────────────────────
    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = build_manifest()
        return self._manifest

    def refresh(self) -> None:
        """Re-introspect (e.g. after a new tool is authored and registered)."""
        self._manifest = None

    @property
    def version(self) -> str:
        return self.manifest["manifest_version"]

    # ── retrieval (Phase 4 agent tools) ─────────────────────────────────────
    def describe_node_kind(self, kind: str) -> dict[str, Any] | None:
        return self.manifest["node_kinds"].get(kind)

    def describe_tool(self, name: str) -> dict[str, Any] | None:
        """Full detail for one tool, including its input JSON schema."""
        entry = next((t for t in self.manifest["tools"] if t["name"] == name), None)
        if entry is None:
            return None
        detail = dict(entry)
        try:
            from neurosurfer.tools.registry import all_tools

            tool = next((t for t in all_tools() if t.name == name), None)
            if tool is not None:
                detail["input_schema"] = tool.schema.input_schema
        except Exception:  # noqa: BLE001 - schema detail is best-effort
            pass
        return detail

    def search_docs(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        return [
            {"path": s.path, "heading": s.heading, "snippet": s.snippet()}
            for s in self._docs.search(query, k=k)
        ]

    # ── injectable context ──────────────────────────────────────────────────
    def render_context(self, *, workflow_tools_only: bool = False) -> str:
        """Compact markdown self-knowledge block for the agent's system prompt.

        ``workflow_tools_only`` lists only workflow-usable (✓) tools — a much
        smaller, focused catalog for design agents (small models lose the plot
        with 30+ tool descriptions they can't assign anyway).
        """
        m = self.manifest
        lines: list[str] = [
            f"# Neurosurfer capabilities (v{m['manifest_version']}, "
            f"neurosurfer {m['neurosurfer_version']})",
            "",
            "## Node kinds",
        ]
        for kind, info in m["node_kinds"].items():
            lines.append(f"- **{kind}** — {info['summary']}")
        ex = m["expressions"]
        lines += [
            "",
            "## Expressions (guards, router cases, break_when, over)",
            f"- functions: {', '.join(ex['functions'])}",
            f"- namespaces: {', '.join(ex['namespaces'])}",
            f"- {ex['guidance']}",
            "",
            ("## Workflow-node tools (the ONLY tools you may assign to nodes)"
             if workflow_tools_only else "## Tools (workflow-usable marked ✓)"),
        ]
        for t in m["tools"]:
            if workflow_tools_only:
                if not t["workflow_usable"]:
                    continue
                desc = (t["description"] or "").split("\n")[0][:160]
                lines.append(f"- `{t['name']}` — {desc}")
                continue
            mark = "✓ " if t["workflow_usable"] else ""
            lines.append(f"- {mark}`{t['name']}` — {t['description']}")
        mcp_servers = m["mcp"]["configured_servers"]
        if mcp_servers:
            names = ", ".join(s["name"] for s in mcp_servers)
            lines += ["", f"## MCP servers configured: {names}"]
        api = m.get("execution_api") or {}
        if api.get("available"):
            lines += [
                "",
                "## Execution API",
                f"- {api['streaming']}",
                "- " + "; ".join(api["endpoints"]),
            ]
        return "\n".join(lines)
