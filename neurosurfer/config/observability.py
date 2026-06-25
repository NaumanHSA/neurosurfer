"""Logging level and the per-run state directory (run transcripts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ObservabilityConfig:
    log_level: str = "INFO"
    state_dir: Path = field(default_factory=lambda: Path.cwd() / ".neurosurfer")

    def transcripts_dir(self) -> Path:
        return self.state_dir / "transcripts"
