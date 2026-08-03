"""Settings models shared by more than one tool.

A tool's `settings_model` is what an *author* configures once, as against
`input_model`, which is what the model fills in per call. See `Tool.settings_model`
for why the two are separate declarations rather than one wider one.

The `format` hints below are the contract with the studio's controls: `directory`
gets a folder picker, `file` gets a file picker. They ride in `json_schema_extra`
so they survive `model_to_schema` and reach every consumer — a front-end, an API
caller reading `/v1/tools/{name}`, and anything generating documentation — rather
than being a rule one renderer happens to know.

**A class docstring here is UI text.** Pydantic puts it in the schema as
`description`, and the schema is what the studio renders above the fields, so the
rationale for a decision goes in a comment and the docstring stays a sentence
somebody would want to read on a panel. Written down because the first version of
this file put nine paragraphs into a settings dialog.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── WorkingDirectory ─────────────────────────────────────────────────────────
#
# Every path the model supplies resolves under `root`, and anything landing
# outside it is refused — see `graph.engine.configured_tools`.
#
# **Required, with a compatibility fallback.** It is `required` because a tool
# that writes files and cannot say where is not configured, and the schema should
# say so. The engine still falls back to the process's working directory when it
# is unset, because every workflow written before this existed depends on that and
# breaking them all is not a bug fix. Validation is what closes the gap: an unset
# root is reported while the workflow is being built, naming the directory the
# files would otherwise land in.


class WorkingDirectory(BaseModel):
    """The folder this step works in."""

    root: str = Field(
        description=(
            "Paths this step is given resolve under this folder, and it cannot "
            "read or write outside it."
        ),
        json_schema_extra={"format": "directory"},
    )
