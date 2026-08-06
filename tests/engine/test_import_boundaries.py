"""What must stay true about *importing* the engine, whatever shape it is in.

These pin three facts that nothing else checks and that no ordinary test would
notice breaking, because a violated import boundary does not fail — it makes the
package slower, or it fails once, at init, on somebody else's machine.

They exist because `executor.py` is being split into a package, and the failure
mode of that kind of move is exactly here: a runner module gains a top-level
`from .node_runner import run_base_node`, the package `__init__` imports the
runner, and suddenly `import neurosurfer.graph` drags in the whole agents stack
at init time. `executor.py` avoids that today with **eight** lazy imports inside
methods, an arrangement that is deliberate, undocumented, and invisible to every
other test in this suite.
"""

from __future__ import annotations

import subprocess
import sys


def _fresh(code: str) -> str:
    """Run *code* in a clean interpreter — `sys.modules` here is already dirty.

    The last line, not the whole output: importing the package prints a startup
    banner, so anything before the answer belongs to the banner.
    """
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1].strip()


def test_importing_the_graph_package_does_not_pull_the_workflow_layer():
    """`graph/__init__.py` promises this in as many words: the engine is a core
    primitive, and the persisted-package layer is loaded on attribute access."""
    out = _fresh(
        "import sys, neurosurfer.graph; "
        "print('neurosurfer.graph.workflow' in sys.modules)"
    )
    assert out == "False"


def test_importing_the_engine_does_not_pull_the_agent_runners():
    """`node_runner` imports `neurosurfer.agents.*`. Every use of it in the
    executor is a lazy import inside a method for this reason, and a split that
    hoists one to module scope would silently undo it."""
    out = _fresh(
        "import sys, neurosurfer.graph.engine; "
        "print('neurosurfer.graph.engine.node_runner' in sys.modules)"
    )
    assert out == "False"


def test_the_executor_is_the_same_object_by_every_path_it_has_ever_had():
    """Three import paths exist in the wild — the package re-export, the engine
    re-export, and the module itself. A split keeps all three or breaks callers
    that a grep of this repo would not find."""
    from neurosurfer.graph import GraphExecutor as by_package
    from neurosurfer.graph.engine import GraphExecutor as by_engine
    from neurosurfer.graph.engine.executor import GraphExecutor as by_module

    assert by_package is by_engine is by_module


def test_the_private_topo_helper_keeps_its_import_path():
    """`_topo_layers` is underscored and in `__all__` at two levels, so it is
    private by name and public by export. It is imported by the studio."""
    from neurosurfer.graph.engine import _topo_layers as by_engine
    from neurosurfer.graph.engine.executor import _topo_layers as by_module

    assert by_engine is by_module
