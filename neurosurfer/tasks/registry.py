"""Task registry — YAML-backed persistent store with a built-in overlay.

User Tasks live in ``~/.neurosurfer/tasks/<name>.yaml`` (or ``tasks_dir`` from
config). Packaged built-in Tasks (code, general, task_builder) ship in the package's
``tasks/builtin/`` directory and are discoverable read-only. On a name conflict the
user Task wins, so a user can override a built-in by saving their own version.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml

from .definition import Provenance, TaskDefinition
from .policy import PolicyCeiling, validate_task

_BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"


class TaskNotFoundError(KeyError):
    pass


class TaskProtectedError(Exception):
    """Raised when a readonly/system task is edited, overridden, or deleted."""


class TaskRegistry:
    def __init__(
        self,
        tasks_dir: Path,
        ceiling: PolicyCeiling | None = None,
        builtin_dir: Path | None = None,
    ) -> None:
        self._dir = tasks_dir
        self._builtin_dir = builtin_dir if builtin_dir is not None else _BUILTIN_DIR
        self._ceiling = ceiling or PolicyCeiling()
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _user_path(self, name: str) -> Path:
        return self._dir / f"{name}.yaml"

    def _builtin_path(self, name: str) -> Path:
        return self._builtin_dir / f"{name}.yaml"

    def _resolve(self, name: str) -> Path | None:
        """User dir takes precedence over the built-in overlay."""
        user = self._user_path(name)
        if user.exists():
            return user
        builtin = self._builtin_path(name)
        if builtin.exists():
            return builtin
        return None

    # ── public API ────────────────────────────────────────────────────────────

    def list(self, *, include_hidden: bool = True) -> list[str]:
        """Names of all registered tasks (user + built-in), sorted and deduped.

        ``include_hidden=False`` drops ``system`` tasks (e.g. task_builder) — the
        user-facing CLI uses that view so internal capabilities stay out of sight.
        """
        names: set[str] = {p.stem for p in self._dir.glob("*.yaml")}
        if self._builtin_dir.is_dir():
            names |= {p.stem for p in self._builtin_dir.glob("*.yaml")}
        ordered = sorted(names)
        if include_hidden:
            return ordered
        return [n for n in ordered if not self._is_hidden(n)]

    def _is_hidden(self, name: str) -> bool:
        try:
            return self.get(name).is_hidden
        except Exception:  # noqa: BLE001 - a parse error must not silently hide a task
            return False

    def path_for(self, name: str) -> Path | None:
        """Resolved YAML path for ``name`` (user dir wins), or None if unknown."""
        return self._resolve(name)

    def is_builtin(self, name: str) -> bool:
        return not self._user_path(name).exists() and self._builtin_path(name).exists()

    def get(self, name: str) -> TaskDefinition:
        path = self._resolve(name)
        if path is None:
            raise TaskNotFoundError(f"Task '{name}' not found in {self._dir} or built-ins")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return TaskDefinition.model_validate(raw)

    def save(self, task: TaskDefinition, *, validate: bool = True) -> None:
        """Persist a user TaskDefinition to the user dir. Validates against the ceiling.

        Only ``user`` tasks live in the user dir; readonly/system tasks ship as
        packaged built-ins and may not be created or overridden at runtime.
        """
        if task.is_protected:
            raise TaskProtectedError(
                f"Only user tasks can be saved; '{task.name}' is kind={task.kind}."
            )
        if self._protected_builtin_exists(task.name):
            raise TaskProtectedError(
                f"'{task.name}' is a protected built-in task and cannot be overridden."
            )
        if validate:
            validate_task(task, self._ceiling)
        data = task.model_dump(mode="json")
        self._user_path(task.name).write_text(
            yaml.dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _protected_builtin_exists(self, name: str) -> bool:
        builtin = self._builtin_path(name)
        if not builtin.exists():
            return False
        try:
            td = TaskDefinition.model_validate(yaml.safe_load(builtin.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            return False
        return td.is_protected

    def delete(self, name: str) -> None:
        """Delete a user Task. Readonly/system tasks are protected and cannot be deleted."""
        try:
            td = self.get(name)
        except TaskNotFoundError:
            raise TaskNotFoundError(f"Task '{name}' not found in {self._dir}") from None
        if td.is_protected:
            raise TaskProtectedError(f"'{name}' is a {td.kind} task and cannot be deleted.")
        path = self._user_path(name)
        if not path.exists():
            raise TaskNotFoundError(f"Task '{name}' not found in {self._dir}")
        path.unlink()

    def clone(self, src: str, dst: str) -> TaskDefinition:
        """Copy ``src`` (built-in or user) into a new editable user task ``dst``.

        The escape hatch for customising a protected built-in: the copy is kind
        ``user`` (editable/deletable), version reset to 1, with fresh provenance.
        """
        source = self.get(src)  # raises TaskNotFoundError if unknown
        if self._resolve(dst) is not None:
            raise ValueError(f"A task named '{dst}' already exists.")
        copy = source.model_copy(update={
            "name": dst,
            "kind": "user",
            "version": 1,
            "provenance": Provenance(
                created_by="clone", source_model=source.provenance.source_model
            ),
        })
        self.save(copy)
        return copy

    def iter(self) -> Iterator[TaskDefinition]:
        for name in self.list():
            yield self.get(name)
