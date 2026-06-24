"""Policy ceiling — hard caps that no authored Task can exceed.

Authored Tasks (hand-written or Task-builder generated) are validated here
before being registered or activated.  The ceiling prevents a Task from
granting itself ungated shell access, widening write scope to the whole
filesystem, or claiming unlimited turns/depth/concurrency.
"""

from __future__ import annotations

from dataclasses import dataclass

from .definition import TaskDefinition

# Paths that write_scope is never allowed to include.
_DENIED_WRITE_ROOTS = {"/", "~", "~/*", ".git", ".git/**", "**/.git/**"}

# Shell policy ordering: tighter policy has higher index.
_SHELL_POLICY_ORDER = {"denied": 2, "readonly": 1, "gated": 0}


@dataclass(frozen=True)
class PolicyCeiling:
    """Immutable hard caps.  Defaults match the plan §4 specifications."""

    max_turns: int = 500
    max_subagent_depth: int = 3
    max_concurrent_subagents: int = 8
    min_shell_policy: str = "gated"   # Task shell_policy must be ≥ this


def _shell_order(policy: str) -> int:
    return _SHELL_POLICY_ORDER.get(policy, 0)


class PolicyViolation(ValueError):
    pass


def validate_task(task: TaskDefinition, ceiling: PolicyCeiling | None = None) -> None:
    """Raise ``PolicyViolation`` if ``task`` exceeds the ceiling.

    Raises on the first violation found (caller can iterate with
    ``validate_task_all`` if it needs all violations at once).
    """
    c = ceiling or PolicyCeiling()
    g = task.guardrails

    if g.max_turns > c.max_turns:
        raise PolicyViolation(
            f"guardrails.max_turns={g.max_turns} exceeds ceiling {c.max_turns}"
        )
    if g.max_subagent_depth > c.max_subagent_depth:
        raise PolicyViolation(
            f"guardrails.max_subagent_depth={g.max_subagent_depth} exceeds ceiling {c.max_subagent_depth}"
        )
    if g.max_concurrent_subagents > c.max_concurrent_subagents:
        raise PolicyViolation(
            f"guardrails.max_concurrent_subagents={g.max_concurrent_subagents} "
            f"exceeds ceiling {c.max_concurrent_subagents}"
        )
    if _shell_order(g.shell_policy) < _shell_order(c.min_shell_policy):
        raise PolicyViolation(
            f"guardrails.shell_policy='{g.shell_policy}' is less restrictive than "
            f"ceiling min '{c.min_shell_policy}'"
        )
    for scope in g.write_scope:
        if scope in _DENIED_WRITE_ROOTS:
            raise PolicyViolation(
                f"guardrails.write_scope contains denied root: '{scope}'"
            )


def validate_task_all(
    task: TaskDefinition, ceiling: PolicyCeiling | None = None
) -> list[str]:
    """Return all policy violation messages (empty list = valid)."""
    errors: list[str] = []
    c = ceiling or PolicyCeiling()
    g = task.guardrails

    if g.max_turns > c.max_turns:
        errors.append(f"max_turns={g.max_turns} exceeds ceiling {c.max_turns}")
    if g.max_subagent_depth > c.max_subagent_depth:
        errors.append(f"max_subagent_depth={g.max_subagent_depth} exceeds ceiling {c.max_subagent_depth}")
    if g.max_concurrent_subagents > c.max_concurrent_subagents:
        errors.append(
            f"max_concurrent_subagents={g.max_concurrent_subagents} exceeds ceiling {c.max_concurrent_subagents}"
        )
    if _shell_order(g.shell_policy) < _shell_order(c.min_shell_policy):
        errors.append(
            f"shell_policy='{g.shell_policy}' less restrictive than min '{c.min_shell_policy}'"
        )
    for scope in g.write_scope:
        if scope in _DENIED_WRITE_ROOTS:
            errors.append(f"write_scope contains denied root: '{scope}'")

    return errors
