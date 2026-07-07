"""Trace-exporter registry.

Resolves which :class:`~.base.TraceExporter` instances are active for this
process. Two ways to configure, matching the plan's enablement decision:

    * **Env-var auto-on** (default) — the first call to :func:`get_active_exporters`
      inspects the environment (see
      :func:`neurosurfer.config.observability.detect_exporters_from_env`) and lazily
      builds the matching adapters. Zero code change.
    * **Explicit code API** — call :func:`configure_exporters` (by name) or
      :func:`register_exporter` (an instance) before the first run to override
      detection.

Missing backend SDKs never break a run: a name that can't be built is warned and
skipped, so a base install (no ``observability`` extra) simply gets no exporters.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .base import MemoryExporter, NullExporter, TraceExporter

logger = logging.getLogger("neurosurfer.observability")

# Builders for backends whose heavy SDKs must stay lazy. Each returns an instance
# or raises ImportError/Exception (caught by the registry → warn + skip).
_BUILDERS: dict[str, Callable[[str], TraceExporter]] = {}


def _build_langfuse(service_name: str) -> TraceExporter:
    from .langfuse import LangfuseExporter

    return LangfuseExporter(service_name=service_name)


def _build_otel(service_name: str) -> TraceExporter:
    from .otel import OtelExporter

    return OtelExporter(service_name=service_name)


_BUILDERS.update({"langfuse": _build_langfuse, "otel": _build_otel})
# Lightweight, always-available exporters (no external SDK).
_SIMPLE: dict[str, Callable[[str], TraceExporter]] = {
    "null": lambda _s: NullExporter(),
    "memory": lambda _s: MemoryExporter(),
}


# ── process-wide state ──────────────────────────────────────────────────────
_active: list[TraceExporter] | None = None  # None ⇒ not yet resolved


def _build(name: str, service_name: str) -> TraceExporter | None:
    builder = _SIMPLE.get(name) or _BUILDERS.get(name)
    if builder is None:
        logger.warning("Unknown trace exporter %r; skipping.", name)
        return None
    try:
        return builder(service_name)
    except ImportError as e:
        logger.warning(
            "Trace exporter %r unavailable (%s). Install `neurosurfer[observability]`.",
            name,
            e,
        )
    except Exception as e:  # noqa: BLE001 — a bad exporter must never break a run
        logger.warning("Failed to initialise trace exporter %r: %s", name, e)
    return None


def configure_exporters(
    names: list[str], *, service_name: str = "neurosurfer"
) -> list[TraceExporter]:
    """Explicitly set the active exporters by name, replacing any prior state.

    Call before the first agent run to override env detection.
    """
    global _active
    _active = [exp for name in names if (exp := _build(name, service_name)) is not None]
    return _active


def register_exporter(exporter: TraceExporter) -> None:
    """Add an already-constructed exporter instance to the active set."""
    global _active
    if _active is None:
        _active = []
    _active.append(exporter)


def get_active_exporters() -> list[TraceExporter]:
    """The active exporters, resolving from the environment on first use."""
    global _active
    if _active is None:
        # Lazy import to avoid a config import cycle at module load.
        from neurosurfer.config.observability import detect_exporters_from_env

        _active = [
            exp
            for name in detect_exporters_from_env()
            if (exp := _build(name, "neurosurfer")) is not None
        ]
    return _active


def reset_exporters() -> None:
    """Clear resolved state (tests re-resolve on next :func:`get_active_exporters`)."""
    global _active
    _active = None


__all__ = [
    "TraceExporter",
    "NullExporter",
    "MemoryExporter",
    "configure_exporters",
    "register_exporter",
    "get_active_exporters",
    "reset_exporters",
]
