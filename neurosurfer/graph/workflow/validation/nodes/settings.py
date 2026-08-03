"""What a tool is configured with, and what happens when it is not.

## The gap these close

A tool's `input_model` says what a *call* takes; its `settings_model` says what an
*author* configures once. Seven workflow-usable tools touch the filesystem and
every one of them needs a directory — and until settings existed there was
nowhere to put one, so `write_file` resolved `path` against whatever working
directory the gateway process happened to have. In a hosted studio that is the
server's own checkout.

Validation is the half that makes the new field matter. A setting nobody filled
in is invisible on a canvas: the node looks complete, the run succeeds, and the
files are simply somewhere else.

## Why an unset root is a warning and not an error

The same reasoning `agent.no_model` records. **Error when nothing can answer,
warning when something can.** Something can: the engine falls back to the process
working directory, which is what every workflow written before this existed
depends on. Erroring would refuse all of them at once — including the repo's own
examples — which is not a bug fix, it is a migration disguised as one.

So the message carries the consequence instead of the severity doing it: it names
the directory the files will actually land in. "Files will be written to
/home/nomi/workspace/neurosurfer" is a sentence that makes somebody act; "root is
required" is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import Severity, ValidationIssue
from ..registry import node_rule

#: Kinds that carry tools. `base` is here for the reason `tools.py` records: the
#: engine gives a base node one round with its tools, so it configures them the
#: same way a react node does.
TOOL_KINDS = ("base", "react", "tool")


def _tool_manifest(tool_name: str) -> Any | None:
    """The registry's static view of one tool, or None if it cannot be read.

    Lazy, like every other registry read in this package, so `workflows` stays
    importable without the full tool registry behind it.
    """
    try:
        from neurosurfer.registry.manifest import manifest_for  # noqa: PLC0415
        from neurosurfer.tools.registry import all_tools  # noqa: PLC0415

        tool = next((t for t in all_tools() if t.name == tool_name), None)
        return None if tool is None else manifest_for(tool)
    except Exception:  # noqa: BLE001 - an unreadable registry checks nothing
        return None


def _configured(node: Any, tool_name: str) -> dict[str, Any]:
    settings = getattr(node, "tool_settings", None) or {}
    value = settings.get(tool_name) if isinstance(settings, dict) else None
    return value if isinstance(value, dict) else {}


def _is_set(value: Any) -> bool:
    """Whether a setting has a value somebody meant to give it.

    A whitespace string is not a directory. Treating `"  "` as configured is how a
    field passes validation and fails at run time, which is the one outcome this
    module exists to prevent.
    """
    return bool(str(value).strip()) if isinstance(value, str) else value is not None


def _label(manifest: Any, tool_name: str) -> str:
    """What to call the tool in a sentence — its title, never its identifier."""
    return str(getattr(manifest, "title", "") or tool_name)


@node_rule(kinds=TOOL_KINDS, severity=Severity.WARNING)
def every_tool_setting_that_is_required_has_a_value(node, ctx, report) -> None:
    """A tool needing a directory, with no directory set."""
    for tool_name in getattr(node, "tools", None) or []:
        manifest = _tool_manifest(tool_name)
        if manifest is None:
            continue  # an unknown tool is `tools_exist`'s to report
        configured = _configured(node, tool_name)
        missing = [s for s in manifest.required_settings if not _is_set(configured.get(s))]
        if not missing:
            continue

        # Named, because the whole point is that the value is not nowhere — it is
        # somewhere nobody chose. A person reading "no directory set" assumes
        # nothing happens; what actually happens is a write into the server's
        # working directory.
        fallback = Path.cwd()
        report.add(ValidationIssue(
            severity=Severity.WARNING,
            kind="tool.setting_missing",
            node_id=node.id,
            subject=tool_name,
            message=(
                f"This step's '{_label(manifest, tool_name)}' has no folder set, "
                f"so it will work in whichever folder the server happens to be "
                f"running from ({fallback})."
            ),
            suggestion="Choose the folder this step should read and write in.",
            detail=f"tool_settings.{tool_name}: missing {', '.join(missing)}",
        ))


@node_rule(kinds=TOOL_KINDS, severity=Severity.WARNING)
def a_configured_folder_exists(node, ctx, report) -> None:
    """A directory that is set, and is not there.

    Checked on the machine validating, which is the machine that will run it in
    every deployment this surface exists for. A typo in a path is otherwise found
    by a run — after the model call upstream of it has been paid for.

    A template is skipped rather than guessed at: `{inputs.workspace}/out` is a
    legitimate root whose value is not known until the run starts.

    Which settings are folders comes from the schema's `format`, not from a list
    kept here — the same declaration the studio's folder picker reads. A second
    list would be a second vocabulary for one fact, and the one that is wrong is
    always the one nobody is looking at.
    """
    for tool_name in getattr(node, "tools", None) or []:
        manifest = _tool_manifest(tool_name)
        if manifest is None or manifest.settings_schema is None:
            continue
        configured = _configured(node, tool_name)
        props = manifest.settings_schema.get("properties") or {}

        for name, prop in props.items():
            if not isinstance(prop, dict) or prop.get("format") != "directory":
                continue
            raw = configured.get(name)
            if not isinstance(raw, str) or not raw.strip() or "{" in raw:
                continue

            path = Path(raw).expanduser()
            if path.is_dir():
                continue
            # Two different mistakes, and the difference is worth a sentence: a
            # missing folder is usually a typo, a file where a folder was
            # expected is usually the wrong path entirely.
            exists_as_file = path.exists()
            report.add(ValidationIssue(
                severity=Severity.WARNING,
                kind="tool.setting_unknown_folder",
                node_id=node.id,
                subject=tool_name,
                message=(
                    f"This step points '{_label(manifest, tool_name)}' at {raw}, "
                    f"which is a file, not a folder."
                    if exists_as_file
                    else f"This step points '{_label(manifest, tool_name)}' at "
                         f"{raw}, which does not exist on the server."
                ),
                suggestion=(
                    "Pick the folder that contains it."
                    if exists_as_file
                    else "Check the path, or create the folder before running."
                ),
                detail=f"tool_settings.{tool_name}.{name} = {raw!r}",
            ))


@node_rule(kinds=TOOL_KINDS, severity=Severity.WARNING)
def every_configured_setting_belongs_to_its_tool(node, ctx, report) -> None:
    """A setting the tool does not have.

    The same shape as `binding.unused_argument` and for the same reason: the
    engine does the right thing at run time — `ConfiguredTool` reads only the
    field the tool declares as its root — but it cannot tell you that a name
    matched nothing, which is a value somebody configured and no tool will ever
    read.
    """
    settings = getattr(node, "tool_settings", None) or {}
    if not isinstance(settings, dict):
        return

    for tool_name, values in settings.items():
        if not isinstance(values, dict) or not values:
            continue
        manifest = _tool_manifest(str(tool_name))
        if manifest is None or manifest.settings_schema is None:
            continue  # unknown, or a tool we cannot read — neither is this rule's
        declared = set(manifest.settings_schema.get("properties") or {})
        for name in values:
            if name in declared:
                continue
            have = ", ".join(sorted(declared)) or "nothing"
            report.add(ValidationIssue(
                severity=Severity.WARNING,
                kind="tool.unknown_setting",
                node_id=node.id,
                subject=str(tool_name),
                message=(
                    f"This step sets '{name}' on "
                    f"'{_label(manifest, str(tool_name))}', which has no such "
                    f"setting, so it will be ignored."
                ),
                suggestion=f"It can be configured with: {have}.",
                detail=f"tool_settings.{tool_name}.{name}",
            ))


@node_rule(kinds=TOOL_KINDS, severity=Severity.WARNING)
def settings_are_configured_for_a_tool_the_node_holds(node, ctx, report) -> None:
    """Settings left behind for a tool that has since been detached.

    Removing a tool from a node does not remove what it was configured with, and
    it should not: detaching and re-attaching while you try two approaches would
    otherwise throw the directory away every time. But settings for a tool that is
    no longer here read as configuration when they are residue, and the next
    person cannot tell the difference.
    """
    settings = getattr(node, "tool_settings", None) or {}
    if not isinstance(settings, dict):
        return
    held = set(getattr(node, "tools", None) or [])
    for tool_name, values in settings.items():
        if not isinstance(values, dict) or not values or str(tool_name) in held:
            continue
        report.add(ValidationIssue(
            severity=Severity.WARNING,
            kind="tool.setting_for_absent_tool",
            node_id=node.id,
            subject=str(tool_name),
            message=(
                f"This step is configured for '{tool_name}', which is not one of "
                f"its tools."
            ),
            suggestion="Attach it, or clear the leftover configuration.",
            detail=f"tool_settings.{tool_name}",
        ))
