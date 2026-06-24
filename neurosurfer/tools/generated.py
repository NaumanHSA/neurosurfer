"""Dynamic, on-disk tool registry for Architect-generated tools (Phase E3).

Built-in tools are a static list in :mod:`neurosurfer.tools.registry`. Tools the
Architect authors at build time (Phase E4–E6) instead live as standalone ``.py``
files under ``~/.neurosurfer/tools/`` and are discovered at runtime by this module,
then folded into :func:`neurosurfer.tools.registry.all_tools`.

On-disk layout::

    ~/.neurosurfer/tools/
        <name>.py     ← module defining a single `Tool` subclass
        <name>.json   ← provenance sidecar (generated_by, created_at, source_workflow)

Security note: importing a file here executes its module body. Nothing is written to
this directory without the explicit user-approval gate (Phase E5); discovery therefore
only ever loads code the user has approved.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from ..config.paths import generated_tools_dir as _default_generated_tools_dir
from .base import Tool

logger = logging.getLogger(__name__)

__all__ = [
    "GeneratedToolsConfig",
    "GeneratedToolMeta",
    "load_generated_tools",
    "save_generated_tool",
    "list_generated_tools",
    "delete_generated_tool",
]


@dataclass
class GeneratedToolMeta:
    """Provenance for a generated tool (companion ``<name>.json``)."""

    name: str
    description: str = ""
    generated_by: str = "architect"
    source_workflow: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "generated_by": self.generated_by,
            "source_workflow": self.source_workflow,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneratedToolMeta:
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            generated_by=d.get("generated_by", "architect"),
            source_workflow=d.get("source_workflow"),
            created_at=d.get("created_at", datetime.now(UTC).isoformat()),
        )


@dataclass
class GeneratedToolsConfig:
    """Paths for the on-disk generated-tools directory."""

    dir: Path = field(default_factory=_default_generated_tools_dir)

    def tool_path(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.py"

    def meta_path(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.json"


# Cache loaded modules by stable name → (file mtime, module). Re-exec only when the
# file on disk changes, so frequent all_tools() calls stay cheap.
_MODULE_CACHE: dict[str, tuple[float, ModuleType]] = {}


# ── loading ─────────────────────────────────────────────────────────────────────

def load_generated_tools(cfg: GeneratedToolsConfig | None = None) -> list[Tool]:
    """Discover and instantiate every generated tool. Never raises — bad files are
    logged and skipped so one broken tool can't take down the whole registry."""
    cfg = cfg or GeneratedToolsConfig()
    if not cfg.dir.is_dir():
        return []

    tools: list[Tool] = []
    seen: set[str] = set()
    for py in sorted(cfg.dir.glob("*.py")):
        if py.stem.startswith("_"):
            continue
        try:
            module = _import_file(py)
        except Exception as exc:  # noqa: BLE001 - isolate broken generated tools
            logger.warning("Skipping generated tool %s: import failed: %s", py.name, exc)
            continue
        for inst in _instantiate_tools(module):
            if not inst.name or inst.name in seen:
                continue
            seen.add(inst.name)
            tools.append(inst)
    return tools


def _instantiate_tools(module: ModuleType) -> list[Tool]:
    found: list[Tool] = []
    for obj in vars(module).values():
        if not (inspect.isclass(obj) and issubclass(obj, Tool) and obj is not Tool):
            continue
        # Only classes actually defined in this module (skip imported base classes).
        if obj.__module__ != module.__name__:
            continue
        if inspect.isabstract(obj):
            continue
        try:
            found.append(obj())
        except Exception as exc:  # noqa: BLE001 - a tool that won't construct is skipped
            logger.warning("Generated tool class %s failed to instantiate: %s", obj.__name__, exc)
    return found


def _import_file(py: Path) -> ModuleType:
    resolved = py.resolve()
    key = hashlib.md5(str(resolved).encode()).hexdigest()[:8]
    mod_name = f"neurosurfer._generated_tools.{py.stem}_{key}"
    mtime = resolved.stat().st_mtime

    cached = _MODULE_CACHE.get(mod_name)
    if cached and cached[0] == mtime:
        return cached[1]

    spec = importlib.util.spec_from_file_location(mod_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[mod_name] = (mtime, module)
    return module


# ── persistence (used by the E5 approval gate) ────────────────────────────────────

def save_generated_tool(
    code: str,
    meta: GeneratedToolMeta,
    cfg: GeneratedToolsConfig | None = None,
) -> Path:
    """Write a generated tool's source + provenance sidecar to disk. Returns the .py path."""
    cfg = cfg or GeneratedToolsConfig()
    cfg.dir.mkdir(parents=True, exist_ok=True)
    py = cfg.tool_path(meta.name)
    py.write_text(code, encoding="utf-8")
    cfg.meta_path(meta.name).write_text(
        json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
    )
    return py


def list_generated_tools(cfg: GeneratedToolsConfig | None = None) -> list[GeneratedToolMeta]:
    """Return provenance for every generated tool that has a sidecar."""
    cfg = cfg or GeneratedToolsConfig()
    if not cfg.dir.is_dir():
        return []
    metas: list[GeneratedToolMeta] = []
    for meta_file in sorted(cfg.dir.glob("*.json")):
        try:
            metas.append(GeneratedToolMeta.from_dict(json.loads(meta_file.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue
    return metas


def delete_generated_tool(name: str, cfg: GeneratedToolsConfig | None = None) -> bool:
    """Remove a generated tool's .py and .json. Returns True if anything was removed."""
    cfg = cfg or GeneratedToolsConfig()
    removed = False
    for p in (cfg.tool_path(name), cfg.meta_path(name)):
        if p.exists():
            p.unlink()
            removed = True
    return removed


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")
