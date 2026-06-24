"""Loose-output coercion helpers for ``register_*`` tools.

Small / local models phrase tool arguments loosely — synonyms for enum values,
alternative key names, stringified numbers. These helpers normalise that output so
a single phrasing slip does not fail an entire registration. Shared by
``register_task`` (and the ``register_*`` tools added in later pillars) so every
authoring tool is equally forgiving.
"""

from __future__ import annotations

from typing import Any

# Shell access levels, with the loose phrasings local models tend to emit.
SHELL_POLICY_ALIASES: dict[str, str] = {
    "denied": "denied", "deny": "denied", "none": "denied", "off": "denied",
    "no": "denied", "disabled": "denied",
    "readonly": "readonly", "read_only": "readonly", "read-only": "readonly",
    "ro": "readonly", "read": "readonly",
    "gated": "gated", "gate": "gated", "ask": "gated", "approve": "gated",
    "prompt": "gated", "on": "gated",
}


def coerce_enum(value: Any, aliases: dict[str, str], default: str) -> str:
    """Map a loosely-phrased string onto a known enum value, else ``default``.

    Non-string input always yields ``default`` (a bare key the model invented).
    """
    if isinstance(value, str):
        return aliases.get(value.strip().lower(), default)
    return default


def coerce_str_list(value: Any) -> list[str]:
    """Accept ``None`` / a bare string / a list and return a clean ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return value


def fill_key(data: Any, primary: str, alts: tuple[str, ...]) -> Any:
    """If ``primary`` is absent from ``data``, copy it from the first present alt.

    Returns the original object when ``data`` is not a dict or ``primary`` is
    already present; otherwise a shallow copy with ``primary`` filled in. Used to
    accept alternative argument key names from loose model output.
    """
    if not isinstance(data, dict) or primary in data:
        return data
    out = dict(data)
    for alt in alts:
        if alt in out:
            out[primary] = out[alt]
            break
    return out
