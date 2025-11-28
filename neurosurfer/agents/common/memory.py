# neurosurfer/agents/common/memory.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Mapping, List


@dataclass
class MemorySlot:
    key: str
    value: Any
    scope: str = "ephemeral"  # "ephemeral" | "persistent"
    description: str = ""     # short human description
    visible_to_llm: bool = True
    created_by: str | None = None  # tool name, etc.


@dataclass
class MemorySnapshot:
    """
    Immutable view of the current memory state, typically used when
    preparing kwargs for a tool call.

    Combines persistent and ephemeral scopes.
    """
    persistent: Dict[str, Any] = field(default_factory=dict)
    ephemeral: Dict[str, Any] = field(default_factory=dict)

    def as_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten into a single dict for tool kwargs.

        Ephemeral keys override persistent keys if they collide.
        """
        merged = dict(self.persistent)
        merged.update(self.ephemeral)
        return merged


class AgentMemory:
    """
    Unified memory abstraction for agents and tools.

    It separates TWO scopes:

    1) Persistent memory:
       - Long(er)-lived state for this agent instance / run.
       - Typical contents:
           - user_id, thread_id
           - files_context, workdir
           - db_engine, rag_agent, config flags
           - any other objects that should survive across tool calls.
       - Cleared only when you explicitly reset the agent or call `clear_persistent`.

    2) Ephemeral memory:
       - Short-lived scratch space for passing small items between steps.
       - Typical contents:
           - `extras` from a single tool call:
               e.g. rag_context, generated_plots, intermediate results
       - Cleared after each tool execution (by the agent),
         then repopulated from `ToolResponse.extras`.

    Usage pattern in an agent (e.g. ReActAgent):

        # Before first tool call
        memory.set_persistent("user_id", 123)
        memory.set_persistent("thread_id", 456)
        memory.set_persistent("files_context", {...})

        # Building kwargs for a tool:
        kwargs = {**tool_call.inputs, **memory.snapshot_for_tool().as_flat_dict()}
        tool_response = tool(**kwargs)

        # After tool call:
        memory.clear_ephemeral()
        memory.update_from_extras(tool_response.extras)  # store extras in ephemeral

    You can also selectively store some extras into persistent scope if needed.
    """

    def __init__(self) -> None:
        self._persistent: Dict[str, Any] = {}
        self._ephemeral: Dict[str, Any] = {}
        self._slots: Dict[str, MemorySlot] = {}

    # -------- Persistent API --------
    def set_persistent(self, key: str, value: Any, description: str = "", visible_to_llm: bool = True, created_by: str | None = None) -> None:
        self._slots[key] = MemorySlot(
            key=key,
            value=value,
            scope="persistent",
            description=description,
            visible_to_llm=visible_to_llm,
            created_by=created_by,
        )

    # def set_persistent(self, key: str, value: Any) -> None:
    #     """
    #     Set a persistent key. Overwrites existing value if present.
    #     """
    #     self._persistent[key] = value

    def get_persistent(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a value from persistent scope.
        """
        return self._persistent.get(key, default)

    def remove_persistent(self, key: str) -> None:
        """
        Remove a persistent key if it exists.
        """
        self._persistent.pop(key, None)

    def clear_persistent(self) -> None:
        """
        Clear ALL persistent memory.
        """
        self._persistent.clear()

    # -------- Ephemeral API --------
    def set_ephemeral(self, key: str, value: Any, description: str = "", visible_to_llm: bool = True, created_by: str | None = None) -> None:
        self._slots[key] = MemorySlot(
            key=key,
            value=value,
            scope="ephemeral",
            description=description,
            visible_to_llm=visible_to_llm,
            created_by=created_by,
        )

    # def set_ephemeral(self, key: str, value: Any) -> None:
    #     """
    #     Set an ephemeral key (per-step scratch).
    #     """
    #     self._ephemeral[key] = value

    def get_ephemeral(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a value from ephemeral scope.
        """
        return self._ephemeral.get(key, default)

    def remove_ephemeral(self, key: str) -> None:
        """
        Remove an ephemeral key if it exists.
        """
        self._ephemeral.pop(key, None)

    def clear_ephemeral(self) -> None:
        """
        Clear ALL ephemeral memory.
        """
        self._slots = {k: v for k, v in self._slots.items() if v.scope != "ephemeral"}
        # self._ephemeral.clear()

    # -------- Combined / convenience --------
    def snapshot_for_tool(self) -> List[MemorySlot]:
        """All slots (ephemeral + persistent) that are runtime-available."""
        return list(self._slots.values())

    # def snapshot_for_tool(self) -> MemorySnapshot:
    #     """
    #     Take a snapshot of the current memory state for tool invocation.

    #     This is the main entry point for agents when they want to pass memory
    #     into tools as kwargs.
    #     """
    #     return MemorySnapshot(
    #         persistent=dict(self._persistent),
    #         ephemeral=dict(self._ephemeral),
    #     )

    def update_from_extras(self, extras: Dict[str, Any], scope: str = "ephemeral", created_by: str | None = None) -> None:
        """
        Convention: extras can be either bare values or small dicts with metadata:
          "schema": {"value": {...}, "description": "schema for students.csv"}
        """
        for key, raw in extras.items():
            if isinstance(raw, dict) and "value" in raw:
                value = raw["value"]
                description = raw.get("description", "")
                visible_to_llm = bool(raw.get("visible_to_llm", True))
            else:
                value = raw
                description = ""
                visible_to_llm = False  # default: runtime-only unless annotated

            if scope == "persistent":
                self.set_persistent(
                    key,
                    value,
                    description=description,
                    visible_to_llm=visible_to_llm,
                    created_by=created_by,
                )
            else:
                self.set_ephemeral(
                    key,
                    value,
                    description=description,
                    visible_to_llm=visible_to_llm,
                    created_by=created_by,
                )

    # def update_from_extras(
    #     self,
    #     extras: Mapping[str, Any],
    #     *,
    #     scope: str = "ephemeral",
    # ) -> None:
    #     """
    #     Update memory from a tool's `extras` dict.

    #     By default, extras go into the ephemeral scope (i.e. valid for the next step).
    #     You can choose `scope="persistent"` if you want a tool to store
    #     long-lived state.
    #     """
    #     if not extras:
    #         return

    #     if scope not in ("ephemeral", "persistent"):
    #         raise ValueError("scope must be 'ephemeral' or 'persistent'")

    #     target = self._ephemeral if scope == "ephemeral" else self._persistent
    #     for k, v in extras.items():
    #         target[k] = v

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Convenience accessor:

        1) Check ephemeral (highest priority),
        2) then persistent.
        """
        if key in self._ephemeral:
            return self._ephemeral[key]
        return self._persistent.get(key, default)

    def as_debug_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        For debugging/logging: shows both scopes without exposing object reprs too deeply.
        """
        def _safe_repr_map(src: Dict[str, Any]) -> Dict[str, str]:
            out: Dict[str, str] = {}
            for k, v in src.items():
                try:
                    out[k] = repr(v)
                except Exception:
                    out[k] = f"<unreprable {type(v).__name__}>"
            return out

        return {
            "persistent": _safe_repr_map(self._persistent),
            "ephemeral": _safe_repr_map(self._ephemeral),
        }

    # -------- Legacy-ish helpers --------
    def set(self, key: str, value: Any, *, persistent: bool = False) -> None:
        """
        Shorthand:
        - persistent=False → ephemeral
        - persistent=True → persistent
        """
        if persistent:
            self.set_persistent(key, value)
        else:
            self.set_ephemeral(key, value)

    def items(self) -> Dict[str, Any]:
        """
        Flattened view (ephemeral overrides persistent), mainly for older code
        that expects `memory.items()` to be passed to tools.
        """
        return self.snapshot_for_tool().as_flat_dict()


    def resolve_keys(self, keys: List[str]) -> Dict[str, Any]:
        """Return a dict of {key: value} for the given slot keys."""
        out: Dict[str, Any] = {}
        for k in keys:
            slot = self._slots.get(k)
            if slot is not None:
                out[k] = slot.value
        return out

    def llm_visible_summary(self) -> str:
        """
        Short textual listing of memory for the ReAct prompt.
        Does NOT leak raw values; only names + descriptions.
        """
        lines = []
        for slot in self._slots.values():
            if not slot.visible_to_llm:
                continue
            scope = "ephemeral" if slot.scope == "ephemeral" else "persistent"
            desc = slot.description or "(no description)"
            created = f" (from tool: {slot.created_by})" if slot.created_by else ""
            lines.append(f"- key: {slot.key} [{scope}]{created}\n  description: {desc}")
        return "\n".join(lines) if lines else "(no memory slots yet)"