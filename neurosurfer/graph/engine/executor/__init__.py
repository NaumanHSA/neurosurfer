"""Running a graph: the scheduler, and one runner per node kind.

## Why this is a package

It was one 1,950-line module, and it was going to keep growing — every new node
kind adds a runner, and eleven kinds had put eleven of them in the same file as
the scheduler that dispatches to them.

The split is by **what runs**, not by layer: `routing.py` holds everything about
choosing a branch, `iteration.py` everything about running a body more than
once. A defect in how a `map` fans out is now in the file called `iteration`,
rather than nine hundred lines into a file called `executor`.

## The import rule that shapes it

`node_runner` imports `neurosurfer.agents.*`, and the engine is a core primitive
that must stay importable without the agent stack — so **every** use of it in
this package is a lazy import inside the function that needs it, exactly as it
was in the single module. Hoisting one to the top of a submodule would pull the
agents in at package-init time, since this file imports the submodules.

`tests/engine/test_import_boundaries.py` fails if that happens. It is the only
thing that would notice.

## The runners are functions, not methods

Each takes the executor as its first argument (`ex`) rather than being a method
on it. Two reasons, one of them mechanical:

- a runner needs perhaps six things off the executor, and passing it explicitly
  says so, where `self` says only "everything";
- **a missed `self.` becomes a lint error rather than a runtime one.** Ruff's
  F821 flags an undefined name in a module-level function, which turned moving
  fourteen methods out of a class into a check a machine could complete.

`GraphExecutor` keeps its method surface — `_run_map_node` and friends are thin
forwarders — because they are part of how the class reads and how its subclasses
and tests reach it.
"""

from .core import GraphExecutor, _topo_layers

__all__ = ["GraphExecutor", "_topo_layers"]
