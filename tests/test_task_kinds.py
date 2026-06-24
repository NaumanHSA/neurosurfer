"""Task classification: kind (user/readonly/system) — CLI visibility + protection."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.tasks.definition import TaskDefinition
from neurosurfer.tasks.registry import TaskProtectedError, TaskRegistry


def _user_task(name: str = "my-task", **kw) -> TaskDefinition:
    return TaskDefinition(name=name, system_prompt="do it", tools=["finish"], **kw)


def _reg(tmp_path: Path) -> TaskRegistry:
    return TaskRegistry(tmp_path / "tasks")  # empty user dir → built-in overlay only


# ── kind field + properties ──────────────────────────────────────────────────
def test_default_kind_is_user():
    td = _user_task()
    assert td.kind == "user"
    assert td.is_protected is False
    assert td.is_hidden is False


def test_protected_and_hidden_properties():
    assert _user_task(kind="readonly").is_protected is True
    assert _user_task(kind="readonly").is_hidden is False
    sysd = _user_task(kind="system")
    assert sysd.is_protected is True
    assert sysd.is_hidden is True


# ── built-in classifications ─────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["code", "general"])
def test_capability_builtins_are_readonly(tmp_path, name):
    td = _reg(tmp_path).get(name)
    assert td.kind == "readonly"
    assert td.is_protected and not td.is_hidden


def test_visible_builtin_lineup_is_exactly_code_and_general(tmp_path):
    visible = set(_reg(tmp_path).list(include_hidden=False))
    assert visible == {"code", "general"}
    assert "doc_gen" not in visible
    assert "code_understanding" not in visible
    assert "memory_curator" not in visible  # the curator builder stays hidden


@pytest.mark.parametrize("name", ["task_builder", "memory_curator"])
def test_builder_metaagents_are_system(tmp_path, name):
    td = _reg(tmp_path).get(name)
    assert td.kind == "system"
    assert td.is_protected and td.is_hidden


# ── hidden filtering ─────────────────────────────────────────────────────────
def test_list_hides_system_but_get_still_works(tmp_path):
    reg = _reg(tmp_path)
    assert "task_builder" in reg.list()                          # discoverable internally
    assert "task_builder" not in reg.list(include_hidden=False)  # hidden from the CLI
    assert "code" in reg.list(include_hidden=False)              # readonly stays visible
    assert reg.get("task_builder").name == "task_builder"        # still runnable internally


# ── protection ───────────────────────────────────────────────────────────────
def test_delete_protected_builtins_refused(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(TaskProtectedError):
        reg.delete("code")
    with pytest.raises(TaskProtectedError):
        reg.delete("task_builder")


def test_save_refuses_protected_kind(tmp_path):
    with pytest.raises(TaskProtectedError):
        _reg(tmp_path).save(_user_task(name="x", kind="readonly"))


def test_save_refuses_shadowing_protected_builtin(tmp_path):
    with pytest.raises(TaskProtectedError):
        _reg(tmp_path).save(_user_task(name="code"))  # would shadow the readonly built-in


def test_user_task_saves_and_deletes(tmp_path):
    reg = _reg(tmp_path)
    reg.save(_user_task(name="mine"))
    assert "mine" in reg.list()
    reg.delete("mine")
    assert "mine" not in reg.list()


# ── path_for ─────────────────────────────────────────────────────────────────
def test_path_for_builtin_and_user(tmp_path):
    reg = _reg(tmp_path)
    builtin_path = reg.path_for("code")
    assert builtin_path is not None and builtin_path.name == "code.yaml"
    assert "builtin" in str(builtin_path)
    reg.save(_user_task(name="mine"))
    assert reg.path_for("mine") == tmp_path / "tasks" / "mine.yaml"
    assert reg.path_for("nope") is None


# ── clone ────────────────────────────────────────────────────────────────────
def test_clone_builtin_into_editable_user_task(tmp_path):
    reg = _reg(tmp_path)
    clone = reg.clone("code", "my-code-agent")
    assert clone.kind == "user"
    assert clone.version == 1
    assert clone.provenance.created_by == "clone"
    assert clone.tools == reg.get("code").tools  # behaviour copied
    assert "my-code-agent" in reg.list(include_hidden=False)
    reg.delete("my-code-agent")  # now an ordinary editable task (no raise)


def test_clone_to_existing_name_refused(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(ValueError):
        reg.clone("code", "general")  # collides with a built-in
    reg.save(_user_task(name="taken"))
    with pytest.raises(ValueError):
        reg.clone("general", "taken")


# ── CLI pure ops ─────────────────────────────────────────────────────────────
def test_cli_op_clone_and_delete(tmp_path):
    from neurosurfer.app.cli.commands.task import op_clone, op_delete

    reg = _reg(tmp_path)
    assert "general-copy" in op_clone(reg, "general", "general-copy")
    assert "general-copy" in reg.list()
    assert "Deleted" in op_delete(reg, "general-copy")
    assert "general-copy" not in reg.list()
