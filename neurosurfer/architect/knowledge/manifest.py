"""Auto-derived capability manifest (Phase 3).

Builds a structured, versioned description of what neurosurfer can actually do —
node kinds and their fields, the expression language, the tool catalog, configured
MCP servers, the workflow package format, and the execution API — by introspecting
the live code, never by hand-maintained lists that can drift.

The one deliberately hand-written part is the per-kind *guidance* text in
``_KIND_GUIDANCE`` (what each node kind is for and which fields matter). The
freshness test asserts its coverage equals ``_VALID_NODE_KINDS``, so adding a new
kind without updating the guidance fails CI — drift is impossible, staleness is
loud.

``manifest_version`` is a short content hash: any change to the derived
capabilities yields a new version, so agents/UIs can display and compare
"built against capability set vX".
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

__all__ = ["build_manifest", "manifest_version"]


# Hand-written guidance per node kind. COVERAGE IS TESTED against
# _VALID_NODE_KINDS — extend this dict whenever a kind is added to the engine.
_KIND_GUIDANCE: dict[str, dict[str, Any]] = {
    "base": {
        "summary": "One LLM call. For writing/summarising/classifying/transforming text.",
        "key_fields": ["purpose", "goal", "expected_result", "mode", "output_schema",
                       "writes", "when", "policy"],
        "requires": ["purpose or goal"],
    },
    "react": {
        "summary": "An LLM that calls tools in a loop. For filesystem, search, web, shell work.",
        "key_fields": ["purpose", "goal", "tools", "writes", "when", "policy"],
        "requires": ["tools"],
    },
    "function": {
        "summary": "Deterministic Python: imports and calls `callable` with inputs + dep outputs.",
        "key_fields": ["callable", "when", "writes", "on_error"],
        "requires": ["callable"],
    },
    "python": {
        "summary": "Alias of function (imports and calls a Python callable).",
        "key_fields": ["callable", "when", "writes", "on_error"],
        "requires": ["callable"],
    },
    "tool": {
        "summary": "Directly invokes one registered tool with tool_args + inputs (no LLM).",
        "key_fields": ["tools", "tool_args", "when", "writes", "on_error"],
        "requires": ["tools (first entry is invoked)"],
    },
    "router": {
        "summary": "Selects ONE downstream branch; non-selected targets are pruned. "
                   "Simple form: `routes` ({label: target}) — the router itself "
                   "classifies via one LLM call instructed by purpose/goal, with "
                   "`repair` retrying an invalid answer before `default`. Advanced "
                   "form: `cases` ([{when, to}]) — deterministic predicates, no LLM "
                   "call.",
        "key_fields": ["routes", "repair", "cases", "default", "purpose", "depends_on"],
        "requires": ["routes or cases (not both); every target must depends_on the router"],
    },
    "loop": {
        "summary": "Repeats a nested `body` sub-graph, always capped by "
                   "`max_iterations`. Stop condition: `until` (plain-English, judged "
                   "by an internal LLM decision each iteration; a CONTINUE verdict's "
                   "reason reaches the next iteration as {feedback}) or `break_when` "
                   "(deterministic expression, no LLM call) — not both. `accumulate` "
                   "collects each iteration's output into a list.",
        "key_fields": ["body", "max_iterations", "until", "break_when", "accumulate", "as"],
        "requires": ["body", "max_iterations >= 1", "until XOR break_when (or neither)"],
    },
    "map": {
        "summary": "Runs a nested `body` once per item of the collection from the `over` "
                   "expression, `concurrency` at a time. Output is the ordered list of "
                   "per-item results (implicit gather).",
        "key_fields": ["body", "over", "as", "concurrency"],
        "requires": ["body", "over"],
    },
    "subgraph": {
        "summary": "Runs a nested `body` sub-graph once (composition). Its final outputs "
                   "become this node's output.",
        "key_fields": ["body", "body_outputs"],
        "requires": ["body"],
    },
    "input": {
        "summary": "Human-in-the-loop pause. Resolves from a pre-supplied input/var named "
                   "by `writes` (or the node id) — the API resume path — else asks "
                   "interactively; otherwise the run finishes as awaiting_input.",
        "key_fields": ["purpose", "options", "writes"],
        "requires": ["purpose (the question)"],
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
    from neurosurfer.graph.engine.schema import _VALID_NODE_KINDS

    kinds: dict[str, dict[str, Any]] = {}
    for kind in sorted(_VALID_NODE_KINDS):
        guidance = _KIND_GUIDANCE.get(kind)
        if guidance is None:
            # A kind exists in the engine with no written guidance. Surface it
            # explicitly (the freshness test turns this into a hard failure).
            guidance = {"summary": "UNDOCUMENTED KIND — update _KIND_GUIDANCE",
                        "key_fields": [], "requires": []}
        kinds[kind] = dict(guidance)
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
        out.append({
            "name": t.name,
            "description": t.description,
            "inputs": props,
            "workflow_usable": t.name in wf_names,
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
