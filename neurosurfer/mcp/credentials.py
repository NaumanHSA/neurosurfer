"""What a server needs, against what this machine already has.

A registry entry declares its requirements; that is a statement about the server,
not about whether *you* can run it. Those are different questions and only the
second one should ever block a build. Conflating them is how the Architect came to
refuse a workflow over `Authorization` — a header whose declared value is
``Bearer {smithery_api_key}``, a placeholder naming a key already stored in
settings, with the code to fill it already written in `_remote_headers`.

Two ideas do the work:

**Ask for the placeholder, not the slot.** A templated credential's *name* is
where the value goes (`Authorization`); the thing a person can actually supply is
the placeholder inside it (`smithery_api_key`). A checklist that says "provide
Authorization" is asking for something nobody possesses under that name.

**Availability is ambient and account-scoped.** Like the discovery engine, the
credentials this account holds are resolved once on the request thread and bound
for the work — an Architect build runs on a raw thread and inherits no context of
its own.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from .registry import CredentialRequirement

__all__ = [
    "asked_name",
    "blocking",
    "known_credentials",
    "lookup",
    "remember_credential",
    "satisfied_from",
    "supply_for",
    "use_credentials",
]

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][\w.-]*)\}")
_PROVIDED: ContextVar[Mapping[str, str] | None] = ContextVar("mcp_credentials", default=None)


def known_credentials() -> Mapping[str, str]:
    """Credentials bound for this context (account settings, secrets)."""
    return _PROVIDED.get() or {}


@contextmanager
def use_credentials(values: Mapping[str, str] | None) -> Iterator[None]:
    """Bind *values* as the account's known credentials for the duration."""
    if not values:
        yield
        return
    token = _PROVIDED.set(dict(values))
    try:
        yield
    finally:
        _PROVIDED.reset(token)


def remember_credential(name: str, value: str) -> None:
    """Add *name* to the credentials bound for this context.

    A build binds its account's credentials **once**, on the request thread,
    because the worker runs on a raw thread with an empty context. That snapshot
    is taken before the build starts — so a value supplied *during* it, from the
    verification secrets prompt, is written to the account store and is still
    invisible to the very next lookup.

    Seen live: the prompt asked for seven values, the user supplied them, the
    requirement check immediately reported all seven missing, and it asked again.
    """
    if not name or not str(value).strip():
        return
    current = _PROVIDED.get()
    if current is None:
        # No binding to extend (a CLI build, or a test). Starting one here is
        # enough — `use_credentials` would have reset it, and there is nothing
        # to reset to.
        _PROVIDED.set({name: str(value)})
        return
    current[name] = str(value)


def lookup(name: str) -> tuple[str, str]:
    """``(value, source)`` for *name*, or ``("", "")``.

    Account-bound values win over the process environment: a user who set
    something in Settings means it, and should not be silently overruled by
    whatever the gateway happened to be started with.

    Names are matched exactly first, then case-insensitively — a registry
    placeholder is conventionally `smithery_api_key` while the environment
    variable holding the same value is `SMITHERY_API_KEY`, and treating those as
    two different credentials would ask the user for one they have already set.
    """
    if not name:
        return "", ""
    provided = known_credentials()
    for store, source in ((provided, "account"), (os.environ, "environment")):
        if name in store and str(store[name]).strip():
            return str(store[name]), source
        lowered = name.lower()
        for key, value in store.items():
            if key.lower() == lowered and str(value).strip():
                return str(value), source
    return "", ""


def asked_name(req: CredentialRequirement) -> str:
    """What a person would have to supply for *req*.

    For a templated value that is the placeholder inside it; the requirement's own
    name is the slot the value goes into, which is not something anyone has.
    """
    match = _PLACEHOLDER.search(req.template or "")
    return match.group(1) if match else req.name


def satisfied_from(req: CredentialRequirement) -> str:
    """Where *req* is already met from — ``account``, ``environment``, or ``""``."""
    return lookup(asked_name(req))[1]


def blocking(reqs: list[CredentialRequirement]) -> list[CredentialRequirement]:
    """The required credentials that are *not* already available.

    This — not `required` — is what decides whether a build can proceed and what a
    checklist should ask for.
    """
    return [r for r in reqs if r.required and not satisfied_from(r)]


def supply_for(reqs: list[CredentialRequirement]) -> dict[str, dict[str, str]]:
    """Values to hand an install, split by where the entry puts them.

    Returns ``{"env": {...}, "headers": {...}}`` keyed by the name the install
    path expects: a templated header is filled through its *placeholder*, which is
    the key `_remote_headers` substitutes on.
    """
    out: dict[str, dict[str, str]] = {"env": {}, "headers": {}}
    for req in reqs:
        key = asked_name(req)
        value, _ = lookup(key)
        if not value:
            continue
        bucket = "headers" if req.where == "header" else "env"
        out[bucket][key] = value
    return {k: v for k, v in out.items() if v}
