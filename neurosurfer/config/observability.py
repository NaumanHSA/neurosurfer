"""Logging level, the per-run state directory (run transcripts), and which
trace exporters ship agent runs to an external monitoring backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
