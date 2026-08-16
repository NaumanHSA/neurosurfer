"""Killing a timed-out child on a platform without process groups.

`os.killpg`, `os.getpgid` and `signal.SIGKILL` are POSIX-only. On Windows the
missing name raises `AttributeError` — which is **not** an `OSError`, so it
escaped the `except (ProcessLookupError, PermissionError, OSError)` and the
function blew up before reaching its own fallback. A timed-out `run_command` or
`python_exec` child was simply left running, and three of the documented Windows
test failures were this.

Simulated rather than skipped: deleting the attributes reproduces the platform
difference exactly, so the guard is testable on Linux instead of being asserted
and hoped for.
"""

from __future__ import annotations

import os
import signal

import pytest

from neurosurfer.registry.core.system.python_exec.sandbox import (
    _kill_group as sandbox_kill,
)
from neurosurfer.registry.core.system.run_command import _kill_group as command_kill

BOTH = pytest.mark.parametrize(
    "kill_group", [command_kill, sandbox_kill], ids=["run_command", "python_exec"]
)


@BOTH
def test_it_does_not_raise_where_there_are_no_process_groups(kill_group, monkeypatch):
    """The Windows shape: no `killpg`, no `SIGKILL`, and a pid that is not ours."""
    monkeypatch.delattr(os, "killpg", raising=False)
    monkeypatch.delattr(os, "getpgid", raising=False)
    monkeypatch.delattr(signal, "SIGKILL", raising=False)

    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

    kill_group(4242)                       # must not raise

    assert killed == [(4242, signal.SIGTERM)], (
        "it should still kill the child, using the signal the platform has"
    )


@BOTH
def test_it_still_kills_the_group_where_there_is_one(kill_group, monkeypatch):
    """POSIX is unchanged: the group call is preferred and `os.kill` is not reached."""
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    monkeypatch.setattr(os, "kill", lambda pid, sig: pytest.fail("fell through to os.kill"))

    kill_group(4242)

    assert calls == [(4242, signal.SIGKILL)]


@BOTH
def test_a_dead_process_is_not_an_error(kill_group, monkeypatch):
    """The race this function exists inside: the child may exit between the
    timeout firing and the kill landing."""
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    def gone(*_a, **_k):
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", gone)
    monkeypatch.setattr(os, "kill", gone)

    kill_group(4242)                       # must not raise
