"""R2: the Architect (builder + conversation) is usable from pure Python.

Guards the structural invariant that the workflow builder does not depend on the
CLI — you can drive it from code without importing anything under ``neurosurfer.app.cli``.
"""

from __future__ import annotations

import sys


def test_architect_public_api_importable():
    from neurosurfer.architect import ArchitectBuilder, ArchitectConversation

    assert ArchitectBuilder.__name__ == "ArchitectBuilder"
    assert ArchitectConversation.__name__ == "ArchitectConversation"


def test_importing_architect_pulls_in_no_cli():
    # Drop any already-imported cli modules, then import the architect fresh and
    # assert it did not drag the CLI in transitively.
    for name in [m for m in list(sys.modules) if m.startswith("neurosurfer.app.cli")]:
        del sys.modules[name]

    import importlib

    import neurosurfer.architect as arch

    importlib.reload(arch)

    cli_loaded = [m for m in sys.modules if m.startswith("neurosurfer.app.cli")]
    assert cli_loaded == [], f"architect pulled in CLI modules: {cli_loaded}"


def test_conversation_lives_under_workflows_not_cli():
    from neurosurfer.architect.conversation import ArchitectConversation

    assert ArchitectConversation.__module__ == "neurosurfer.architect.conversation"
