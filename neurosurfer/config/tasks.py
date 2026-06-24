"""Task registry location."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TasksConfig:
    dir: Path = field(default_factory=lambda: Path.home() / ".neurosurfer" / "tasks")
