"""Stored values a node may use, and the rule that keeps them out of prompts.

The whole design serves one constraint: **a secret reaches the tool and never the
model.** A database password interpolated into a node's goal is in the request, in
the trace, in the run record, and in whatever the trace is exported to — and none
of those are places anyone decided to put it.

So secrets live outside the interpolation scope entirely. `{NAME}` cannot resolve
to one because they were never in the mapping the prompt is rendered against; the
only way to reach a value is `${NAME}` inside `tool_args`, on a node that named it
in `secrets:`. Two different syntaxes for two different destinations, and the one
that reaches a secret is not the one that reaches a prompt.

Values come from the credential context the run binds (see
:mod:`neurosurfer.mcp.credentials`), so this module needs no account awareness.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["expand_node_secrets", "redact", "secret_refs"]

_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Below this, a "secret" is short enough that redacting it would blank out
# ordinary text that merely matches — a value of "1" would turn every digit in a
# trace into a mask. Such a value is not protecting anything anyway.
_MIN_REDACT = 6


def secret_refs(text: str) -> list[str]:
    """The `${NAME}` references in *text*."""
    return _REF.findall(text or "")


def _values() -> dict[str, str]:
    from neurosurfer.mcp.credentials import known_credentials

    return dict(known_credentials())


def expand_node_secrets(text: str, node: Any) -> str:
    """Fill `${NAME}` in *text* for names *node* declared in `secrets:`.

    An undeclared name is left exactly as written rather than expanded: the
    declaration is the authorisation, so honouring a reference the node never
    claimed would make `secrets:` documentation instead of a gate.
    """
    if not text or "${" not in text:
        return text
    allowed = set(getattr(node, "secrets", None) or ())
    if not allowed:
        return text
    values = _values()

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in allowed and name in values:
            return values[name]
        return m.group(0)

    return _REF.sub(_sub, text)


def redact(value: Any) -> Any:
    """*value* with any bound secret replaced by a mask.

    Applied to what gets traced, not to what gets sent: the tool is given the real
    value and the record of the call is not. Strings only — a secret that reached
    a non-string field is a different bug and silently rewriting it would hide it.
    """
    secrets = [v for v in _values().values() if isinstance(v, str) and len(v) >= _MIN_REDACT]
    if not secrets:
        return value
    return _walk(value, secrets)


def _walk(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            if secret in value:
                value = value.replace(secret, "••••••")
        return value
    if isinstance(value, dict):
        return {k: _walk(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, secrets) for v in value]
    return value
