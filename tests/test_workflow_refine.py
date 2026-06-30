"""Tests for the self-healing refinement loop (Phase E7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from neurosurfer.architect.refine import WorkflowRefiner
from neurosurfer.graph.workflow.package import load_package

# ── doubles ──────────────────────────────────────────────────────────────────────

class _NR:
    def __init__(self, error=None, skipped=False):
        self.error = error
        self.skipped = skipped


class _Result:
    def __init__(self, nodes):
        self.nodes = nodes


class _StubRunner:
    """Returns canned results in order (last one repeats)."""

    def __init__(self, results):
        self.results = results
        self.calls = 0

    def run(self, pkg, inputs, **kw):
        r = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return r


class _StubRegistry:
    def __init__(self, pkg_dir: Path):
        self.pkg_dir = pkg_dir

    def get(self, name):
        return load_package(self.pkg_dir)


class _Resp:
    def __init__(self, text):
        self._t = text

    def text(self):
        return self._t


class _FakeProvider:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        self.calls += 1
        return _Resp(self.reply)


def _make_pkg(pkg_dir: Path) -> None:
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "workflow.yaml").write_text(
        yaml.dump({"name": pkg_dir.name, "version": "0.1.0", "entrypoint": "graph.yaml"}),
        encoding="utf-8",
    )
    graph = {
        "name": pkg_dir.name,
        "description": "test wf",
        "nodes": [{"id": "n", "kind": "function", "callable": "os:getcwd"}],
        "outputs": ["n"],
    }
    (pkg_dir / "graph.yaml").write_text(yaml.dump(graph), encoding="utf-8")


def _patch_reply(patch: dict) -> str:
    return json.dumps({"diagnosis": "needs a tweak", "patch": patch})


def _refiner(pkg_dir, results, reply, max_rounds=3) -> WorkflowRefiner:
    return WorkflowRefiner(
        _FakeProvider(reply),
        registry=_StubRegistry(pkg_dir),
        runner=_StubRunner(results),
        max_rounds=max_rounds,
    )


# ── tests ────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clean_run_needs_no_patch(tmp_path):
    _make_pkg(tmp_path / "wf")
    refiner = _refiner(tmp_path / "wf", [_Result({"n": _NR()})], _patch_reply({"purpose": "x"}))
    res = await refiner.refine("wf", {})
    assert res.ok
    assert res.patched_nodes == []
    assert refiner.provider.calls == 0  # never consulted the doctor


@pytest.mark.asyncio
async def test_fail_then_patch_then_succeed(tmp_path):
    pkg_dir = tmp_path / "wf"
    _make_pkg(pkg_dir)
    results = [_Result({"n": _NR(error="boom")}), _Result({"n": _NR()})]
    refiner = _refiner(pkg_dir, results, _patch_reply({"purpose": "fixed up"}))
    res = await refiner.refine("wf", {})
    assert res.ok
    assert res.patched_nodes == ["n"]
    # patch landed in the agents override layer
    override = yaml.safe_load((pkg_dir / "agents" / "n.yaml").read_text())
    assert override["purpose"] == "fixed up"


@pytest.mark.asyncio
async def test_gives_up_after_max_rounds(tmp_path):
    pkg_dir = tmp_path / "wf"
    _make_pkg(pkg_dir)
    results = [_Result({"n": _NR(error="always broken")})]  # never recovers
    refiner = _refiner(pkg_dir, results, _patch_reply({"purpose": "try"}), max_rounds=2)
    res = await refiner.refine("wf", {})
    assert not res.ok
    assert "Still failing after 2 rounds" in res.message


@pytest.mark.asyncio
async def test_no_patch_proposed_stops(tmp_path):
    pkg_dir = tmp_path / "wf"
    _make_pkg(pkg_dir)
    results = [_Result({"n": _NR(error="boom")}), _Result({"n": _NR()})]
    refiner = _refiner(pkg_dir, results, _patch_reply({}))  # empty patch
    res = await refiner.refine("wf", {})
    assert not res.ok
    assert "No node could be patched" in res.message


@pytest.mark.asyncio
async def test_invalid_patch_is_rejected(tmp_path):
    pkg_dir = tmp_path / "wf"
    _make_pkg(pkg_dir)
    results = [_Result({"n": _NR(error="boom")}), _Result({"n": _NR()})]
    # patch wires a tool that doesn't exist → re-validation must fail
    refiner = _refiner(pkg_dir, results, _patch_reply({"tools": ["nonexistent_xyz"]}))
    res = await refiner.refine("wf", {})
    assert not res.ok
    assert "invalid workflow" in res.message


@pytest.mark.asyncio
async def test_patch_only_keeps_allowed_fields(tmp_path):
    pkg_dir = tmp_path / "wf"
    _make_pkg(pkg_dir)
    results = [_Result({"n": _NR(error="boom")}), _Result({"n": _NR()})]
    # 'id' and 'kind' are not patchable; 'purpose' is
    reply = _patch_reply({"purpose": "ok", "id": "hacked", "kind": "react"})
    refiner = _refiner(pkg_dir, results, reply)
    res = await refiner.refine("wf", {})
    assert res.ok
    override = yaml.safe_load((pkg_dir / "agents" / "n.yaml").read_text())
    assert override["purpose"] == "ok"
    assert override["id"] == "n"  # not overwritten by the patch
    assert "kind" not in override or override["kind"] != "react"
