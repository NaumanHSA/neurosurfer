"""Auto-derived capability manifest (Phase 3).

Builds a structured, versioned description of what neurosurfer can actually do —
node kinds and their fields, the expression language, the tool catalog, configured
MCP servers, the workflow package format, and the execution API — by introspecting
the live code, never by hand-maintained lists that can drift.

Two layers, with different ways of staying honest:

*Schema* — kinds, fields, tools, expressions, the API. Introspected, and pinned by
freshness tests that compare the manifest against the live engine.

*Craft* — ``_KIND_GUIDANCE`` (what each kind is for), ``_BUILD_RULES`` (how builds
actually go wrong) and ``_RECIPES`` (worked shapes). This cannot be introspected,
so it decays differently and needs a different guard: ``_KIND_GUIDANCE`` coverage
is asserted against ``_VALID_NODE_KINDS``, and every recipe is **validated through
the real** ``validate_package``. A recipe describing a node the engine would reject
fails the suite, which is what stops this layer drifting into fiction the way
prose documentation does.

This layer exists because ``docs/`` is written for humans and lags the code, and
retrieval over it was answering build questions with release notes. Agent
knowledge lives here, next to what it describes.

``manifest_version`` is a short content hash: any change to the derived
capabilities yields a new version, so agents/UIs can display and compare
"built against capability set vX".
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

__all__ = ["build_manifest", "manifest_version", "_BUILD_RULES", "_RECIPES"]


# Hand-written guidance per node kind: what this kind is *for*, and what has gone
# wrong with it before. COVERAGE IS TESTED against _VALID_NODE_KINDS — extend this
# dict whenever a kind is added to the engine.
#
# Structure is NOT written here. `key_fields`, `requires`, `calls_model` and
# `terminal` are derived from `graph.engine.kinds` in `_derive_node_kinds()`,
# because they were a restatement of something the engine already knows and they
# drifted: this dict called `python` a callable-importing node while the studio
# called the same kind an inline-code node, and nothing compared either claim to
# the executor. Say what a kind is *for* here; the engine says what it *takes*.
_KIND_GUIDANCE: dict[str, dict[str, Any]] = {
    "base": {
        "summary": "One LLM call, for writing/summarising/classifying/transforming "
                   "text. It CANNOT take tools — it only sees its prompt. A step "
                   "that must touch the outside world is a `tool` or `react` node.",
        "notes": [
            "Structured output is `mode: structured` + `output_schema` (an import "
            "path to a pydantic model). Without it a base node returns text, so a "
            "node asked for an object emits JSON *as a string* and anything "
            "checking the shape fails. A build blocked over exactly this, "
            "reporting that structured output 'is not available with the current "
            "node schema' — it is, and nothing had told it.",
        ],
    },
    "react": {
        "summary": "An LLM that calls tools, for any step that must touch the "
                   "outside world AND needs a model to decide what to send. That "
                   "covers multi-step filesystem/search/web/shell work, and also "
                   "the single call whose arguments must be composed — the SQL for "
                   "a query tool, the phrase for a search tool, the body for an "
                   "API tool.",
        "notes": [
            "This is the ONLY kind that both reasons and acts. `base` reasons and "
            "cannot act; `tool` acts and cannot reason. A step needing both is a "
            "react node — splitting it into a base node that writes an instruction "
            "and a tool node that was supposed to read it does not work, because "
            "nothing passes the instruction to the tool.",
            "`tool_args` here are BOUND arguments, not the whole call: the engine "
            "supplies them on every call the model makes and removes them from the "
            "schema it is offered. This is how a react node uses a credential — "
            "`secrets: [DB_URL]` with `tool_args: {dsn: '${DB_URL}'}` means the "
            "model composes only the query, and never sees the connection string.",
        ],
    },
    "function": {
        "summary": "Deterministic Python: imports and calls `callable` with inputs + dep outputs.",
    },
    "python": {
        "summary": "Alias of function (imports and calls a Python callable).",
    },
    "tool": {
        "summary": "Directly invokes ONE registered tool. Use it when every "
                   "required argument is a constant or an interpolation of an "
                   "input/upstream output (`read_file` with "
                   "`tool_args: {path: '{doc}'}`).",
        "notes": [
            "A stored value (database password, connection string, API key) is "
            "reached by naming it in `secrets: [NAME]` and writing `${NAME}` in "
            "`tool_args`. It is filled at call time and never enters a prompt.",
            "NEVER write ${NAME} into purpose/goal/expected_result — those reach "
            "the model, and validation refuses it. If a step needs a credential, "
            "it is a `tool` node, not a `base` one.",
        ],
    },
    "router": {
        "summary": "Selects ONE downstream branch; non-selected targets are pruned. "
                   "`routes` ({label: target}) classifies with one LLM call; "
                   "`cases` ([{when, to}]) evaluates predicates and costs nothing.",
    },
    "loop": {
        "summary": "Repeats a nested `body` sub-graph until a stop condition holds. "
                   "A CONTINUE verdict from `until` reaches the next iteration as "
                   "{feedback}; `accumulate` collects each output into a list.",
    },
    "map": {
        "summary": "Runs a nested `body` once per item of the collection from the `over` "
                   "expression, `concurrency` at a time. Output is the ordered list of "
                   "per-item results (implicit gather).",
    },
    "subgraph": {
        "summary": "Runs a nested `body` sub-graph once (composition). Its final outputs "
                   "become this node's output.",
    },
    "input": {
        "summary": "Human-in-the-loop pause. Resolves from a pre-supplied input/var named "
                   "by `writes` (or the node id) — the API resume path — else asks "
                   "interactively; otherwise the run finishes as awaiting_input.",
    },
    "output": {
        "summary": "What the graph returns. With `value` set it is an interpolated "
                   "template over graph inputs, upstream outputs and `writes` vars; "
                   "without one it passes its single dependency through unchanged, "
                   "preserving its type.",
    },
}

# ── the craft layer ──────────────────────────────────────────────────────────
#
# Everything above says what the engine *is*: kinds, fields, tools, expressions,
# all introspected from live code and pinned by freshness tests. This says how to
# USE it, which cannot be introspected — and is where every observed build failure
# has actually come from.
#
# The rule for adding to this: **one entry per failure seen in a real transcript.**
# Not what a node kind might be misused for in principle. Speculative guidance
# costs context on every build forever and protects against nothing, and the
# always-on block is the most expensive real estate in the system.
#
# Each rule cites the build that earned it, so a future reader can decide whether
# it still applies rather than treating the list as scripture.
_BUILD_RULES: list[dict[str, str]] = [
    {
        "rule": "A `tool` node has NO model. If you cannot fill `tool_args` "
                "completely right now, the step is a `react` node with that tool "
                "attached.",
        "why": "A tool node's goal text is read by nobody — there is no model call "
               "in it. Writing an instruction there and expecting the tool to "
               "follow it produces a call with missing arguments.",
        "seen": "sales-report build: 7 nodes with tools=['query_sql'], no "
                "tool_args, and a prose goal describing the SQL to write. Every "
                "one failed with an opaque error from the server.",
    },
    {
        "rule": "A credential is NEVER a graph input. Name it in `secrets: [NAME]` "
                "on the node that needs it and write `${NAME}` inside `tool_args`.",
        "why": "Graph inputs are in the interpolation scope, so an input named "
               "`db_password` is reachable from every node's prompt — and from "
               "there the model's context, the trace, and any exporter. Secrets "
               "live outside that scope entirely and reach only the tool call.",
        "seen": "sales-report build: `db_connection_secret: object` declared as a "
                "graph input and interpolated into a node's goal.",
    },
    {
        "rule": "Check the tool can do the thing, not just that its name matches. "
                "If nothing in the catalog can, say so — do not attach the closest "
                "name.",
        "why": "A node that validates with a plausible-but-wrong tool is worse "
               "than a blocked build: it goes green and produces a file that is "
               "the wrong format, or an empty result presented as an answer.",
        "seen": "sales-report build: `assemble_pdf_report` was given `write_file` "
                "— a text writer — after no PDF server was installed.",
    },
    {
        "rule": "A step that reads or writes anything outside the model is "
                "external, however ordinary it sounds. 'Query the orders table' is "
                "external; so is 'save the report'.",
        "why": "Steps marked internal never reach the capability ladder, so "
               "nothing checks whether a tool for them exists. The error surfaces "
               "much later, as a node that cannot run.",
        "seen": "sales-report build: 6 steps whose intent began 'Query …' were "
                "planned as internal; the plan claimed 2 external steps of 16.",
    },
    {
        "rule": "Finish by calling a terminal tool. A graph that validates is not "
                "a build that ended.",
        "why": "Stopping after the last node leaves the build hanging until a "
               "nudge restarts it, and each nudge spends turns that the rest of "
               "the build then does not have.",
        "seen": "sales-report build: 5 early stops, 2 of them after validation "
                "had already reported ok.",
    },
]

# Worked node shapes, as real graphs. Every one is validated by
# ``tests/test_architect_knowledge.py`` through the same ``validate_package`` a
# build runs, so a recipe cannot describe a node the engine would reject — if the
# engine changes, the test fails here and the guidance is corrected with the code.
# That is the whole point: prose about how to build cannot be introspected, so it
# is written as something executable instead.
#
# Scoped by the same rule as _BUILD_RULES: a kind earns a recipe when a real build
# got it wrong. `loop`, `map`, `router`, `subgraph`, `function` and `input` have no
# recipe because no observed failure involved them — their `summary` and
# `key_fields` have been sufficient. Add one when that stops being true.
_RECIPES: dict[str, dict[str, Any]] = {
    "tool": {
        "title": "every argument known → a tool node, no model",
        "graph": {
            "name": "recipe_tool",
            "inputs": [{"name": "doc_path", "type": "file"}],
            "nodes": [
                {
                    "id": "read_doc",
                    "kind": "tool",
                    "tools": ["read_file"],
                    # `read_file` requires `path`; tool_args supplies it whole.
                    "tool_args": {"path": "{doc_path}"},
                    "writes": "doc",
                },
            ],
            "outputs": ["read_doc"],
        },
    },
    "tool_with_secret": {
        "title": "a credential reaches the tool and never a prompt",
        "graph": {
            "name": "recipe_tool_secret",
            "nodes": [
                {
                    "id": "fetch_items",
                    "kind": "tool",
                    "tools": ["http"],
                    # Declared here, substituted at call time. It is not in the
                    # interpolation scope, so no prompt can reach it.
                    "secrets": ["API_TOKEN"],
                    "tool_args": {
                        "url": "https://api.example.com/v1/items",
                        "headers": {"Authorization": "Bearer ${API_TOKEN}"},
                    },
                    "writes": "items",
                },
            ],
            "outputs": ["fetch_items"],
        },
    },
    "react": {
        "title": "an argument must be composed → a react node with the tool",
        "graph": {
            "name": "recipe_react",
            "inputs": [{"name": "topic", "type": "string"}],
            "nodes": [
                {
                    "id": "research",
                    "kind": "react",
                    # The search phrase is not known here — a model has to write
                    # it from {topic}. That is what makes this react, not tool.
                    "tools": ["web_search"],
                    "goal": "Search for recent, credible sources about {topic}. "
                            "Return the three most useful with a one-line note on "
                            "what each contributes.",
                    "writes": "sources",
                },
            ],
            "outputs": ["research"],
        },
    },
    "react_with_secret": {
        "title": "compose the call AND use a credential → bind it on a react node",
        "graph": {
            "name": "recipe_react_secret",
            "inputs": [{"name": "window_months", "type": "integer"}],
            "nodes": [
                {
                    "id": "audit",
                    "kind": "react",
                    "tools": ["sql"],
                    # Bound: the engine supplies `dsn` on every call and hides it
                    # from the schema, so the model composes only `query` and the
                    # connection string never enters its context. Before this, a
                    # step needing both a composed argument and a credential could
                    # not be expressed at all.
                    #
                    # `operation` is deliberately NOT bound. Binding only the
                    # credential leaves the model free to call `list_tables`,
                    # then `table_schema`, then `query` — every operation of the
                    # tool, on one connection it never sees. That is the shape
                    # this goal describes, and it falls out of operations for
                    # free.
                    "secrets": ["AUDIT_DB_URL"],
                    "tool_args": {"dsn": "${AUDIT_DB_URL}"},
                    "goal": "Find the tables holding access activity, then count "
                            "inquiries by result status over the last "
                            "{window_months} months.",
                    "writes": "audit_rows",
                },
            ],
            "outputs": ["audit"],
        },
    },
    "base": {
        "title": "text in, text out, no outside world",
        "graph": {
            "name": "recipe_base",
            "inputs": [{"name": "sources", "type": "string"}],
            "nodes": [
                {
                    "id": "summarise",
                    "kind": "base",
                    "goal": "Write one paragraph for a non-technical reader from "
                            "these notes: {sources}",
                    "writes": "summary",
                },
            ],
            "outputs": ["summarise"],
        },
    },
}

_EXPRESSION_GUIDANCE = (
    "Predicates evaluate against namespaces: inputs.*, nodes.<id> (a node's raw "
    "output), vars.* (explicit `writes`), plus index/item inside loop/map bodies. "
    "Real LLM output arrives with whitespace/case noise — prefer "
    "contains(lower(nodes.x), 'label') over exact equality, or use structured "
    "outputs for exact matching. Missing keys resolve to None (predicates fail "
    "closed, they never crash the run)."
)


def _pydantic_fields(model: type) -> dict[str, dict[str, Any]]:
    """name → {type, default, description, alias} derived from a pydantic model."""
    out: dict[str, dict[str, Any]] = {}
    for name, field in model.model_fields.items():  # type: ignore[attr-defined]
        entry: dict[str, Any] = {
            "type": str(field.annotation).replace("typing.", ""),
            "description": field.description or "",
        }
        if field.alias and field.alias != name:
            entry["alias"] = field.alias
        try:
            default = field.get_default(call_default_factory=True)
            entry["default"] = None if default is None else json.loads(
                json.dumps(default, default=str)
            )
        except Exception:  # noqa: BLE001 - unrepresentable default → omit
            pass
        out[name] = entry
    return out


def _derive_node_kinds() -> dict[str, dict[str, Any]]:
    """Structure from the engine's kind specs, craft from `_KIND_GUIDANCE`.

    `key_fields` and `requires` used to be written out per kind here, beside the
    prose. That is a restatement of something the engine knows, and it drifted
    the way a restatement does: this dict described `python` as taking a
    `callable` while the studio described the same kind as running inline code,
    and neither was checked against the executor.

    So they are derived now. What stays hand-written is the part that genuinely
    cannot be introspected — the summary and the notes, which say *when to reach
    for this kind* and are where every observed build failure got its answer.
    """
    from neurosurfer.graph.engine.kinds import NODE_KIND_SPECS
    from neurosurfer.graph.engine.schema import _VALID_NODE_KINDS

    kinds: dict[str, dict[str, Any]] = {}
    for kind in sorted(_VALID_NODE_KINDS):
        guidance = _KIND_GUIDANCE.get(kind)
        if guidance is None:
            # A kind exists in the engine with no written guidance. Surface it
            # explicitly (the freshness test turns this into a hard failure).
            guidance = {"summary": "UNDOCUMENTED KIND — update _KIND_GUIDANCE"}
        entry = dict(guidance)
        spec = NODE_KIND_SPECS.get(kind)
        if spec is not None:
            entry["key_fields"] = [f.name for f in spec.fields]
            # A requirement is a required field or a stated constraint. Field
            # names are quoted the way the rest of the manifest quotes them, so
            # a model reading this sees the same vocabulary throughout.
            entry["requires"] = [
                f"{f.name} ({f.label.lower()})" for f in spec.required_fields
            ] + list(spec.constraints)
            entry["calls_model"] = spec.calls_model
            entry["terminal"] = spec.terminal
        else:
            entry.setdefault("key_fields", [])
            entry.setdefault("requires", [])
        # Attach the worked shape so `describe_node_kind` answers "what does a
        # correct one look like" as well as "what fields does it take". Kinds
        # without an observed failure have no recipe, by design.
        recipe = _RECIPES.get(kind)
        if recipe is not None:
            entry["recipe"] = recipe["graph"]
        kinds[kind] = entry
    return kinds


def _derive_expressions() -> dict[str, Any]:
    from neurosurfer.graph.engine.expressions import _ALLOWED_FUNCS

    return {
        "functions": sorted(_ALLOWED_FUNCS),
        "operators": [
            "== != < <= > >= in not-in is is-not", "and or not",
            "+ - * / // % ** (bounded)", "x if cond else y", "indexing a[i], a['k']",
        ],
        "namespaces": ["inputs", "nodes", "vars", "state", "index/item (loop & map scope)"],
        "guidance": _EXPRESSION_GUIDANCE,
    }


def _derive_tools() -> list[dict[str, Any]]:
    from neurosurfer.tools.registry import all_tools, workflow_node_tool_names

    wf_names = workflow_node_tool_names()
    out = []
    for t in sorted(all_tools(), key=lambda t: t.name):
        try:
            props = list((t.schema.input_schema or {}).get("properties", {}).keys())
        except Exception:  # noqa: BLE001 - a broken tool schema must not kill the manifest
            props = []
        # The registry's view of the same tool: what it can do, where it runs,
        # and which of its inputs is a credential. These are declared facts, and
        # putting them beside the description is what lets the model stop
        # inferring capability from wording.
        try:
            from neurosurfer.registry import manifest_for

            man = manifest_for(t)
            facts = {
                # What a person calls it, and what it looks like. Neither is for
                # the model — a palette that lists `sql`, `http`, `apply_edit`
                # is showing the engine's vocabulary to somebody who never
                # agreed to learn it. `name` remains the identifier throughout.
                "title": man.title,
                "icon": man.icon,
                "capabilities": sorted(man.capabilities),
                "runtime": man.runtime,
                "secret_inputs": sorted(man.secret_inputs),
                "credential_help": man.credential_help,
                # What an *author* configures once, as against `inputs`, which is
                # what the model fills in per call. `None` for a tool with
                # nothing to configure, and that has to stay distinguishable
                # from an empty object or every tool grows a settings panel with
                # no settings in it.
                "settings_schema": man.settings_schema,
                # Each operation's own schema, so a caller configuring one is
                # shown the fields *it* needs rather than the union of all of
                # them with everything optional. Empty for single-purpose tools,
                # which must not grow an "Operation: [one choice]" selector.
                "operations": [o.to_dict() for o in man.operations],
            }
        except Exception:  # noqa: BLE001 - a manifest fault must not kill the catalog
            facts = {"title": t.name, "icon": "tool", "capabilities": [],
                     "runtime": "in_process", "secret_inputs": [],
                     "credential_help": "", "settings_schema": None,
                     "operations": []}
        out.append({
            "name": t.name,
            "description": t.description,
            "inputs": props,
            "workflow_usable": t.name in wf_names,
            **facts,
        })
    return out


def _derive_mcp() -> dict[str, Any]:
    try:
        from neurosurfer.config.mcp import McpStore

        servers = []
        for cfg in McpStore().list():
            entry: dict[str, Any] = {"name": cfg.name}
            for attr in ("transport", "command", "url", "enabled"):
                val = getattr(cfg, attr, None)
                if val is not None:
                    entry[attr] = val
            servers.append(entry)
        return {"configured_servers": servers}
    except Exception as e:  # noqa: BLE001 - MCP config is optional
        return {"configured_servers": [], "note": f"unavailable: {e}"}


def _derive_package_format() -> dict[str, Any]:
    import dataclasses

    from neurosurfer.graph.engine.schema import Graph
    from neurosurfer.graph.workflow.schema import WorkflowManifest

    return {
        "files": {
            "workflow.yaml": "package manifest (name, version, description, entrypoint, tags)",
            "graph.yaml": "the Graph spec (inputs, nodes, outputs)",
            "agents/<node_id>.yaml": "optional per-node overrides merged over graph.yaml",
        },
        "manifest_fields": [f.name for f in dataclasses.fields(WorkflowManifest)],
        "graph_fields": _pydantic_fields(Graph),
        "input_types": {
            "types": ["string", "integer", "float", "boolean", "object", "array",
                      "file", "image"],
            "notes": [
                "`file` and `image` still arrive as a *path string* — declare one "
                "whenever a step reads a document or looks at a picture, and the "
                "user gets an upload button instead of typing a path that only "
                "resolves on their own machine.",
                "`enum: [a, b]` on a string input renders a choice, which is the "
                "right shape whenever a router downstream switches on the value.",
                "`accept` (e.g. '.csv' or 'image/*') and `max_bytes` narrow a "
                "file input.",
            ],
        },
    }


def _derive_api() -> dict[str, Any]:
    """Enumerate the execution-API routes from a real app instance (best-effort)."""
    try:
        from neurosurfer.app.server.gateway import NeurosurferServer

        app = NeurosurferServer(app_name="manifest-probe").create_app()

        def _walk(routes) -> list:
            # Newer FastAPI/Starlette may nest included routers (lazy proxies with
            # an `original_router`) instead of flattening, so recurse through both
            # `.routes` and `.original_router.routes`.
            out = []
            for r in routes:
                path = getattr(r, "path", None)
                methods = getattr(r, "methods", None)
                if path and methods:
                    out.append((path, methods))
                sub = getattr(r, "routes", None)
                if sub:
                    out.extend(_walk(sub))
                orig = getattr(r, "original_router", None)
                if orig is not None and getattr(orig, "routes", None):
                    out.extend(_walk(orig.routes))
            return out

        endpoints = sorted(
            f"{','.join(sorted(m for m in methods if m != 'HEAD'))} {path}"
            for path, methods in _walk(app.routes)
            if path.startswith(("/v1/workflows", "/v1/runs"))
        )
        return {"available": True, "endpoints": endpoints,
                "streaming": "GET /v1/runs/{run_id}/events is SSE (replay + live tail)"}
    except Exception as e:  # noqa: BLE001 - serve extra may be absent
        return {"available": False, "note": f"gateway not importable: {e}"}


def build_manifest(*, include_api: bool = True) -> dict[str, Any]:
    """Introspect the installed neurosurfer and return the capability manifest."""
    import neurosurfer
    from neurosurfer.graph.engine.schema import GraphNode

    manifest: dict[str, Any] = {
        "neurosurfer_version": neurosurfer.__version__,
        "node_kinds": _derive_node_kinds(),
        "node_fields": _pydantic_fields(GraphNode),
        "build_rules": [dict(r) for r in _BUILD_RULES],
        "expressions": _derive_expressions(),
        "tools": _derive_tools(),
        "mcp": _derive_mcp(),
        "workflow_package": _derive_package_format(),
    }
    if include_api:
        manifest["execution_api"] = _derive_api()
    manifest["manifest_version"] = manifest_version(manifest)
    manifest["generated_at"] = time.time()
    return manifest


def manifest_version(manifest: dict[str, Any]) -> str:
    """Short content hash over the capability sections (volatile fields excluded)."""
    stable = {k: v for k, v in manifest.items()
              if k not in {"generated_at", "manifest_version"}}
    blob = json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
