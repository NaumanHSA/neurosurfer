"""prompt_toolkit completer: typing '/' shows matching commands; a space after a
command offers its subcommands."""

from __future__ import annotations

from prompt_toolkit.completion import Completer, Completion

from .commands.base import CommandRegistry

# Static subcommand hints (command name → [(sub, meta), ...]).
_SUBCOMMANDS: dict[str, list[tuple[str, str]]] = {
    "provider": [
        ("list", "List provider profiles"),
        ("use", "Set the active profile"),
        ("add", "Add a profile"),
        ("delete", "Delete a profile"),
    ],
    "workflow": [
        ("build", "Describe a workflow and let the Architect build it"),
        ("list", "List registered workflows"),
        ("show", "Inspect a workflow's nodes and graph"),
        ("run", "Execute a registered workflow"),
        ("delete", "Remove a workflow from the registry"),
    ],
}


class SlashCompleter(Completer):
    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        body = text[1:]
        if " " not in body:
            # Completing the command name.
            for cmd in self._registry.matches(body):
                yield Completion(
                    cmd.name,
                    start_position=-len(body),
                    display=f"/{cmd.name}",
                    display_meta=cmd.summary,
                )
            return
        # Completing a subcommand.
        name, _, rest = body.partition(" ")
        command = self._registry.get(name)
        if command is None:
            return
        subs = _SUBCOMMANDS.get(command.name, [])
        word = rest.split(" ")[-1]
        for sub, meta in subs:
            if sub.startswith(word):
                yield Completion(sub, start_position=-len(word), display=sub, display_meta=meta)
