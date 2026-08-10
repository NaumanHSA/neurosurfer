"""Logging level, the per-run state directory (run transcripts), and which
trace exporters ship agent runs to an external monitoring backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: What each backend needs before it can be built, as groups of alternatives:
#: every tuple must be satisfied by at least one variable being set.
#:
#: Auto-detection has always used these — an exporter turns *on* because its
#: connection variables are present. `NEUROSURFER_EXPORTERS` used to skip the
#: check entirely, so naming `otel` there built an exporter with no endpoint and
#: the OTel SDK quietly fell back to `http://localhost:4318`, giving a "tracing
#: is off by default" install a live exporter aimed at nothing. An explicit list
#: says *which* backends are wanted, not that they are configured.
EXPORTER_REQUIRED_ENV: dict[str, tuple[tuple[str, ...], ...]] = {
    "langfuse": (("LANGFUSE_PUBLIC_KEY",), ("LANGFUSE_SECRET_KEY",)),
    "otel": (("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),),
}


def missing_exporter_env(name: str, env: dict[str, str] | None = None) -> list[str]:
    """Connection variables *name* needs and does not have.

    Empty when the backend is configured, or when it declares no requirement —
    `memory` and `null` need nothing, and an exporter registered as an instance
    never comes through here.
    """
    env = os.environ if env is None else env
    return [
        " or ".join(group)
        for group in EXPORTER_REQUIRED_ENV.get(name, ())
        if not any(env.get(var) for var in group)
    ]


def detect_exporters_from_env(env: dict[str, str] | None = None) -> list[str]:
    """Which trace exporters to activate, inferred from the environment.

    Auto-on: an exporter turns on when its backend's connection env vars are
    present, so a user gets tracing by exporting keys — no code change. Returns
    an empty list (no-op) when nothing is configured.

    Recognised:
        - ``langfuse`` — when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set.
        - ``otel`` — when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.

    An explicit ``NEUROSURFER_EXPORTERS`` (comma-separated) overrides detection —
    including ``NEUROSURFER_EXPORTERS=none`` / ``""`` to force everything off.
    """
    env = os.environ if env is None else env

    explicit = env.get("NEUROSURFER_EXPORTERS")
    if explicit is not None:
        names = [n.strip().lower() for n in explicit.split(",") if n.strip()]
        return [n for n in names if n not in ("none", "off", "false")]

    exporters: list[str] = []
    if env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"):
        exporters.append("langfuse")
    if env.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        exporters.append("otel")
    return exporters


@dataclass
class ObservabilityConfig:
    log_level: str = "INFO"
    state_dir: Path = field(default_factory=lambda: Path.cwd() / ".neurosurfer")

    # Trace exporters (Langfuse / OpenTelemetry). Empty ⇒ no external tracing.
    # Populated from the environment by ``load_config`` via ``detect_exporters_from_env``.
    exporters: list[str] = field(default_factory=list)
    # Service / project name surfaced to the backend (OTel resource, Langfuse metadata).
    service_name: str = "neurosurfer"

    def transcripts_dir(self) -> Path:
        return self.state_dir / "transcripts"
