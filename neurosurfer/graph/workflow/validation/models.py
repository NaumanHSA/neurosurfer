"""What a validation result *is* — the types every rule and every reader shares.

## Severity is declared, not chosen at the call site

The previous shape had three lists on the report and each check appended to
whichever one it had in mind, at roughly forty different places. "Is a missing
prompt an error or a warning" was therefore not a question you could answer by
reading anything; you had to find the one line that appended it. Severity now
travels **on the issue**, set once where the rule is written, and the report's
`errors` / `warnings` / `infos` are views over a single list.

## `kind` is the rule's identity, `message` is prose

`kind` is a stable identifier — ``tool_gap``, ``agent.no_model`` — and is what
code should match on. `message` is a sentence for a person and may be rewritten
freely; nothing depends on its wording. That separation is what lets the studio
group and filter issues while the text stays plain.

## `detail` exists so the message can stay plain

A validation message reached the UI reading like ``output_schema 'my:Model' does
not import``, which names two things the person never typed. The sentence they
should see and the field name a developer needs are different audiences, so they
are different fields: `message` is what the panel shows, `detail` is the
technical particulars, shown on request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = ["Severity", "ValidationIssue", "ValidationReport", "GAP_KINDS"]


class Severity(StrEnum):
    """How much an issue matters.

    A ``StrEnum`` so a severity serialises as its own name with no encoder, and
    compares equal to the plain string on the wire. It was written as
    ``(str, Enum)``, which behaves the same here and is the form Python retired
    in 3.11 — `NodeMode` in the engine already uses `StrEnum`, so this is the
    codebase agreeing with itself rather than a change of behaviour.
    """

    #: The workflow will not run, or will not run correctly. Blocks registration.
    ERROR = "error"
    #: It runs, but something is probably not what was meant.
    WARNING = "warning"
    #: A suggestion. Nothing is wrong; there may be a better way to say it.
    INFO = "info"


#: Rule ids that mean "no registered tool provides this".
#:
#: Kept as a set rather than folded into the errors list wholesale, because the
#: Architect *routes* these somewhere different — a capability gap goes to the
#: tool-author agent, where a malformed graph is simply wrong. They are errors by
#: severity and gaps by identity, which is exactly what the two views below say.
GAP_KINDS = frozenset({"tool_gap", "capability_gap"})


@dataclass
class ValidationIssue:
    """A single problem found while validating a package."""

    #: Stable rule identity. Match on this, never on `message`.
    kind: str
    #: One sentence, for a person. No field names, no import paths — see `detail`.
    message: str
    severity: Severity = Severity.ERROR
    node_id: str | None = None
    #: What to do about it, in the same plain register as `message`.
    suggestion: str | None = None
    #: The offending name — a missing tool, an unconfigured provider.
    subject: str | None = None
    #: Technical particulars: field names, import paths, the raw parser error.
    detail: str | None = None

    def render(self) -> str:
        where = f"node '{self.node_id}': " if self.node_id else ""
        hint = f" → {self.suggestion}" if self.suggestion else ""
        return f"{where}{self.message}{hint}"


@dataclass
class ValidationReport:
    """Everything validation found, in one list, read through severity views."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    # ── views ────────────────────────────────────────────────────────────────
    #
    # `errors` and `gaps` are complementary rather than nested: a gap is an
    # error by severity, but the two are shown and handled separately, so a
    # reader asking for one should not silently receive the other.

    @property
    def errors(self) -> list[ValidationIssue]:
        """Blocking problems that are *not* missing capabilities."""
        return [
            i for i in self.issues
            if i.severity is Severity.ERROR and i.kind not in GAP_KINDS
        ]

    @property
    def gaps(self) -> list[ValidationIssue]:
        """Blocking problems that are a missing tool."""
        return [
            i for i in self.issues
            if i.severity is Severity.ERROR and i.kind in GAP_KINDS
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.INFO]

    @property
    def ok(self) -> bool:
        """No blocking problem of any kind. Suggestions never block."""
        return not any(i.severity is Severity.ERROR for i in self.issues)

    def summary(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append("Errors:")
            lines += [f"  • {e.render()}" for e in self.errors]
        if self.gaps:
            lines.append("Capability gaps (no registered tool provides this):")
            lines += [f"  • {g.render()}" for g in self.gaps]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  • {w.render()}" for w in self.warnings]
        if self.infos:
            lines.append("Suggestions:")
            lines += [f"  • {i.render()}" for i in self.infos]
        return "\n".join(lines) if lines else "Package is valid."
