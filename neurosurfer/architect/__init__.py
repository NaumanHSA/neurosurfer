"""The Architect — a built-in WorkflowPackage that designs and builds
other WorkflowPackages from a plain-English user intent.

The Architect's graph lives in ``package/`` (a pre-authored WorkflowPackage
that ships with neurosurfer). It is loaded directly from the source tree,
not from the user's registry.

Public API (usable from pure Python — no CLI dependency):

    from neurosurfer.architect import ArchitectBuilder, ArchitectConversation
"""

from __future__ import annotations

from pathlib import Path

from .agent import ArchitectAgent
from .build import ArchitectBuilder, WorkflowInfeasible
from .conversation import ArchitectConversation

_PACKAGE_DIR = Path(__file__).parent / "package"

__all__ = [
    "ArchitectAgent",
    "ArchitectBuilder",
    "ArchitectConversation",
    "WorkflowInfeasible",
]
