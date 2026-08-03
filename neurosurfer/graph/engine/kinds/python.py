"""`python` — today, an alias of `function`.

The executor routes both kinds to the same function and both require an import
path (``executor.py``, ``_run_function_node``), while the studio has described
this one as *"inline Python transformation over state"* since it was added.
There is no inline path; the description is aspirational.

The spec states what the engine does rather than what the label suggests, so the
gap is visible to everything reading it instead of being a discrepancy between a
card and a code path nobody compares. Resolving it — real inline code, or
retiring the kind — is §3.1 of plan 01 and needs a decision, not a patch.
"""

from __future__ import annotations

from . import _common as c
from .function import CALLABLE
from .spec import NodeKindSpec

SPEC = NodeKindSpec(
    kind="python",
    label="Python",
    blurb="Deterministic Python, by import path. Identical to Function today.",
    shape="code",
    data_arrival=("kwargs",),
    fields=(CALLABLE, *c.WIRING),  # no `export` — see function.py
    constraints=(
        "Currently identical to `function` — same executor path, same required "
        "import path. It does not run inline code.",
    ),
)
