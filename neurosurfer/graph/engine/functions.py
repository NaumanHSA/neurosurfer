"""The graph's Python sidecar: `functions: helpers.py`, then bare names in YAML.

## Why a file rather than an import path

A graph is data. It travels as YAML, and YAML cannot hold a Python function —
so anywhere a graph wants real code it has to name it instead. The existing
answer for `function` nodes is a full import path (`nodes.transform:run`), which
works but makes every call site carry module plumbing:

    until: nodes.checks:tagline_is_short

A graph declares its sidecar once, at the top, and then says only what it means:

    functions: helpers.py
    nodes:
      - id: polish
        kind: loop
        until: tagline_is_short        # ← defined in helpers.py

The file sits beside `graph.yaml` and is copied with it on export, so a workflow
package stays self-contained — the same guarantee `nodes/<id>.py` already gives
`function` nodes.

## The part that matters most

It makes a *fact* out of what would otherwise be a guess. `until` accepts either
a plain-English condition or the name of a function, and telling those apart by
inspecting the string — does it contain a dot, a colon, a space? — is the kind
of rule that works until someone writes a condition that looks like an
identifier. Here there is nothing to infer: **the sidecar either defines that
name or it does not.** A one-word English condition and a missing function are
distinguishable, because one of them is in the module and the other is not.

## Loading

The file is imported under a private module name derived from its absolute path,
so two workflows may each have a `helpers.py` without colliding, and it is not
importable as a normal top-level module by accident. Modules are cached by
(path, mtime): editing the sidecar and re-loading the graph picks up the change,
which matters when a notebook holds a graph across edits.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import GraphConfigurationError

__all__ = ["load_sidecar", "resolve_name", "SidecarModule"]

#: (resolved path, mtime_ns) → module
_CACHE: dict[tuple[str, int], Any] = {}


class SidecarModule:
    """A loaded `functions:` file, and the callables it defines."""

    def __init__(self, module: Any, path: Path) -> None:
        self.module = module
        self.path = path

    def get(self, name: str) -> Callable[..., Any] | None:
        """The callable *name* defines, or None if it defines no such callable.

        Non-callable attributes return None deliberately: `until: description`
        matching a module-level string would otherwise be read as a function and
        fail at call time, when the plain-English reading was almost certainly
        meant.
        """
        fn = getattr(self.module, name, None)
        return fn if callable(fn) else None

    def names(self) -> list[str]:
        """Public callables defined here — for error messages that suggest."""
        return sorted(
            n for n in dir(self.module)
            if not n.startswith("_") and callable(getattr(self.module, n, None))
        )


def load_sidecar(spec: str, base_dir: Path | str | None) -> SidecarModule:
    """Import the graph's `functions:` file, relative to the graph's directory.

    Raises `GraphConfigurationError` when the file is missing or will not
    import — both are authoring mistakes worth stopping for, not conditions to
    limp past. A graph that names a sidecar is relying on it.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    path = (base / spec).resolve() if not Path(spec).is_absolute() else Path(spec)

    if not path.is_file():
        raise GraphConfigurationError(
            f"graph declares `functions: {spec}` but no such file exists "
            f"(looked in {base}). It should sit beside the graph file."
        )

    key = (str(path), path.stat().st_mtime_ns)
    if key in _CACHE:
        return SidecarModule(_CACHE[key], path)

    # A private, path-derived name: two packages may both ship `helpers.py`, and
    # neither should shadow the other or become importable as `helpers`.
    mod_name = "_neurosurfer_fns_" + str(abs(hash(str(path))))
    module_spec = importlib.util.spec_from_file_location(mod_name, path)
    if module_spec is None or module_spec.loader is None:
        raise GraphConfigurationError(f"cannot import graph functions file {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[mod_name] = module          # so dataclasses/pickle inside it work
    try:
        module_spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(mod_name, None)
        raise GraphConfigurationError(
            f"graph functions file {path} failed to import: {e}"
        ) from e

    _CACHE[key] = module
    return SidecarModule(module, path)


def resolve_name(sidecar: SidecarModule | None, name: str) -> Callable[..., Any] | None:
    """The callable *name*, or None when it is not a function at all.

    None is the answer that means "read this as plain English", so it is
    returned rather than raised for every not-a-function case — including a
    graph with no sidecar, which can only have meant prose.
    """
    if sidecar is None:
        return None
    return sidecar.get(name)
