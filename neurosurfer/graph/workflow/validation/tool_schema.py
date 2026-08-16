"""The registry's own description of the tools a graph may name.

Split out because both the tool rules and the template walk need it, and putting
it in either would make the other import a module it has no reason to know
about.

Imported lazily inside the functions, as the original did: it keeps the workflows
package importable without the full tool registry behind it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["registered_tool_names", "tool_input_schema"]


def registered_tool_names() -> set[str]:
    """Every tool name the registry knows."""
    from neurosurfer.tools.registry import all_tools  # noqa: PLC0415

    return {t.name for t in all_tools()}


def tool_input_schema(tool_name: str) -> dict[str, Any] | None:
    """The invoked tool's JSON input schema, or None if it cannot be read."""
    try:
        from neurosurfer.tools.registry import all_tools  # noqa: PLC0415

        tool = next((t for t in all_tools() if t.name == tool_name), None)
        if tool is None:
            return None
        return dict(tool.schema.input_schema or {})
    except Exception:  # noqa: BLE001 - a schema we cannot read cannot be checked
        return None
