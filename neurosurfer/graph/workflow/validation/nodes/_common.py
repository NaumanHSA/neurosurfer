"""Rules that hold for a node of any kind.

Whether the provider it names is configured, whether the fields its own kind
declares required are present, and whether the import paths it carries resolve.
None of these is about *what* the node does, which is why they are here rather
than in a per-kind module.
"""

from __future__ import annotations

import difflib

from pydantic import BaseModel

from neurosurfer.graph.engine import import_string
from neurosurfer.graph.engine.json_schema import JsonSchemaError, model_from_json_schema
from neurosurfer.graph.engine.kinds import node_kind_spec

from ..models import Severity, ValidationIssue, ValidationReport
from ..registry import node_rule

#: Required fields whose absence already has a check that says something more
#: useful than "this is missing". The derived check below skips them rather than
#: adding a second, blunter line about the same node.
#:
#: `output_schema` joined them when `agent.structured_without_schema` landed: the
#: two fired together on the same node, and the derived one said it in the
#: engine's vocabulary ("Agent node is missing `output_schema` — …") where the
#: dedicated one says what the author actually did.
_HAS_A_BETTER_CHECK = frozenset({"callable", "tools", "output_schema"})

def _check_provider(node, known: set[str] | None, report: ValidationReport) -> None:
    """A node naming a provider profile that isn't configured.

    Only checked when the caller supplies the configured names — a package is
    validated in plenty of places that have no idea what a given deployment has set
    up, and guessing there would reject valid graphs on a machine that simply hasn't
    been configured yet.
    """
    name = getattr(node, "provider", None)
    if not name or known is None or name in known:
        return
    match = difflib.get_close_matches(name, known, n=1, cutoff=0.7)
    report.add(ValidationIssue(
        severity=Severity.ERROR,
        kind="provider",
        node_id=node.id,
        message=f"This step is set to run on '{name}', which is not set up.",
        detail=f"configured: {', '.join(sorted(known)) or 'none'}",
        suggestion=(f"did you mean '{match[0]}'?" if match
                    else "configure it in Settings, or leave it unset to use the default"),
        subject=name,
    ))


def _check_output_schema(node, report: ValidationReport) -> None:
    path = node.output_schema
    if not path:
        return
    # Written on the node as JSON Schema. Built here so a shape that cannot
    # become a model is reported while the author is still looking at it, rather
    # than as a configuration error partway into a paid run.
    if isinstance(path, dict):
        try:
            model_from_json_schema(path, name=f"{node.id}_output")
        except JsonSchemaError as exc:
            report.add(ValidationIssue(
        severity=Severity.ERROR,
                kind="schema",
                node_id=node.id,
                subject="output_schema",
                message="The shape described for this step's answer is not valid.",
                suggestion="Check the JSON — the detail says which part.",
                detail=str(exc),
            ))
        return
    try:
        obj = import_string(path)
    except Exception as exc:  # noqa: BLE001 - any import failure is a validation error
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="schema",
            node_id=node.id,
            message="The shape this step should return cannot be loaded.",
            suggestion="Check the shape it points at, or describe it here instead.",
            detail=f"output_schema '{path}' does not import ({exc})",
        ))
        return
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="schema",
            node_id=node.id,
            message="What this step points at is not a shape it can return.",
            detail=f"output_schema '{path}' is not a pydantic BaseModel subclass",
        ))


def _check_required_fields(node, report: ValidationReport) -> None:
    """Every field this node's kind marks required, present.

    Derived from ``engine.kinds`` rather than written out per kind, so a kind
    added to that registry is validated here without this file being edited —
    which is the whole point of the specs. It is a floor, not a second opinion:
    ``required`` in a spec means *the engine refuses the node without it*, and
    where something already refuses it with a better message, that message wins
    (see ``_HAS_A_BETTER_CHECK``).
    """
    spec = node_kind_spec(node.kind)
    if spec is None:  # a kind with no spec — the freshness test is what catches that
        return
    for f in spec.required_fields:
        if f.name in _HAS_A_BETTER_CHECK:
            continue
        # A conditional field is required only while its condition holds. An
        # `output_schema` is the whole point of a `structured` node and is
        # meaningless on one returning prose — demanding it either way would make
        # every text node invalid for missing something it is not offered.
        if not f.applies_to(node):
            continue
        value = getattr(node, f.name, None)
        if value is None or (isinstance(value, (str, list, dict, tuple)) and not value):
            report.add(ValidationIssue(
        severity=Severity.ERROR,
                kind="structure",
                node_id=node.id,
                subject=f.name,
                message=f"This {spec.label.lower()} step needs {f.label.lower()} and it is not set.",
                suggestion=f.help or None,
                detail=f"required field `{f.name}` on kind `{spec.kind}`",
            ))


def _check_callable(node, report: ValidationReport) -> None:
    if node.kind not in {"function", "python"}:
        return
    path = node.callable
    if not path:
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="callable",
            node_id=node.id,
            message=f"{node.kind} node has no 'callable' set",
        ))
        return
    try:
        obj = import_string(path)
    except Exception as exc:  # noqa: BLE001 - any import failure is a validation error
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="callable",
            node_id=node.id,
            message=f"callable '{path}' does not import ({exc})",
        ))
        return
    if not callable(obj):
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="callable",
            node_id=node.id,
            message=f"callable '{path}' is not callable",
        ))


# ── rules ────────────────────────────────────────────────────────────────────
#
# Thin wrappers over the checks above. They exist so the *declaration* — which
# kinds, which severity, whether it descends into container bodies — is data the
# registry can be asked about, instead of being implied by where a call sits in
# a hand-written sequence.


# A provider means nothing on a kind that never calls a model.
@node_rule(kinds=("base", "react", "router"), severity=Severity.ERROR)
def provider_is_configured(node, ctx, report) -> None:
    """The provider profile a node names must be one the deployment has."""
    _check_provider(node, ctx.known_providers, report)


@node_rule(kinds=("*",), severity=Severity.ERROR)
def required_fields_present(node, ctx, report) -> None:
    """Every field this node's kind marks required is set."""
    _check_required_fields(node, report)


# Only the kinds that make a model call carry an output shape — see
# `engine/kinds/_common.py`, where `OUTPUT_SCHEMA` belongs to the LLM field set.
# Only `base` carries an output shape — see `engine/kinds/react.py`.
@node_rule(kinds=("base",), severity=Severity.ERROR)
def output_schema_resolves(node, ctx, report) -> None:
    """A declared output shape must be a usable schema."""
    _check_output_schema(node, report)


# `callable` is the whole instruction of a code node and exists on no other kind.
@node_rule(kinds=("function", "python"), severity=Severity.ERROR)
def callable_resolves(node, ctx, report) -> None:
    """A code node's function must import and be callable."""
    _check_callable(node, report)
