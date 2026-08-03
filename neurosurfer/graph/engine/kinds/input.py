"""`input` — the graph pauses and a person answers.

The value is resolved from a pre-supplied input or variable named by ``writes``
(or, failing that, the node id) — which is the path both an API caller and the
studio's chat take. Only when nothing is supplied does the run finish as
``awaiting_input``.

That key is currently a *convention* rather than a declaration: nothing connects
"this graph has an input node called `input`" to "this graph takes an input named
`input`", so the studio reimplements the ``writes or id`` rule in TypeScript to
know what to send. Plan 01 phase 4 makes the node declare it.
"""

from __future__ import annotations

from . import _common as c
from .spec import FieldSpec, NodeKindSpec

INPUT_MODE = FieldSpec(
    name="input_mode",
    type="select",
    label="How it is answered",
    help="`text` takes one free-text message. `dict` collects the named inputs "
         "this workflow declares, each with its own type.",
    options=("text", "dict"),
    group="instruction",
)

SPEC = NodeKindSpec(
    kind="input",
    label="Human Input",
    blurb="Pauses the run and waits for a person to supply a value.",
    shape="io",
    data_arrival=("supplied",),
    # No `instructions`: the conversation happens at run time, in the chat, and a
    # prompt written months earlier on a config panel is not what a person
    # answering reads. The engine still uses it if a graph sets one — it is the
    # question a *headless* CLI run asks — it is simply not something the studio
    # asks an author to fill in.
    #
    # No `options` either. A fixed list of strings was the only structure this
    # kind could express, and it is the wrong one: the answer to "what does this
    # workflow take" is named, typed inputs, not a set of magic strings. Those are
    # declared as graph inputs and edited on this node in `dict` mode.
    #
    # No `writes`. It named the key the value arrived under, and once the inputs
    # are declared that key *is* their name; in `text` mode the node id is the
    # key, which is one fewer name to keep in step.
    fields=(
        INPUT_MODE,
        c.DEPENDS_ON,
        # `when` was kept here longest, on the strength of one example — "only ask
        # for approval when the amount is over the limit". It is a real graph, and
        # the field still could not be explained on the panel: `vars.amount` comes
        # from a `writes` on another node that nothing on screen mentions. An
        # example that needs a second hidden feature to make sense is an argument
        # for building that feature, not for shipping the box. See `_common.py`.
        c.ON_ERROR,
    ),
    constraints=(
        "In `dict` mode the fields a person fills in are the workflow's declared "
        "inputs — add them on this node.",
        "With nothing supplied the run finishes `awaiting_input`, not `failed`; "
        "a client resumes it. Resuming currently re-runs the whole graph, so a "
        "node before this one runs twice.",
    ),
)
