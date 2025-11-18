from __future__ import annotations
from typing import Optional, Union
from pathlib import Path
import logging
import json
from datetime import datetime

from .schema import GraphExecutionResult, GraphNode, NodeExecutionResult
from neurosurfer.agents.agent.responses import StructuredResponse, ToolCallResponse

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Export API
# ------------------------------------------------------------------ #
def export(
    graph_results: GraphExecutionResult,
    export_base_dir: Optional[Union[Path, str]]
) -> None:
    """
    Export node results to disk for nodes that have `export=True`.

    Rules:
    - If node.export_path is a file (has extension) -> use that exact file.
    - If node.export_path is a directory / bare path -> write
        `{node_id}_{timestamp}.ext` inside it.
    - If node.export_path is not set -> write into `export_base_dir`
        (or `base_dir` override) as `{node_id}_{timestamp}.ext`.

    Content:
    - If structured_output is present -> JSON by default.
    - Else if tool_call_output is present -> JSON by default.
    - Else -> raw_output as Markdown by default.

    If the final path has `.json`, will always write JSON.
    If `.md` / `.txt`, will write Markdown (embedding JSON if needed).
    """
    # check if export_dir is not None, and exists
    if export_base_dir is not None:
        export_base_path = Path(export_base_dir)
        if not export_base_path.exists():
            logger.warning("Export base directory does not exist: %s", export_base_dir)
            # make directory
            export_base_path.mkdir(parents=True, exist_ok=True)
            logger.info("Created export base directory: %s", export_base_path)
        elif not export_base_path.is_dir():
            logger.warning("Export base path exists but is not a directory: %s. Falling back to 'exports' directory.", export_base_dir)
            export_base_path = Path("exports")
            export_base_path.mkdir(parents=True, exist_ok=True)

    graph = graph_results.graph
    # node_map() lets us look up GraphNode by id
    node_map = graph.node_map()

    for node_id, node_result in graph_results.nodes.items():
        node: Optional[GraphNode] = node_map.get(node_id)
        if not node:
            logger.warning("No GraphNode found for node_id %s; skipping export.", node_id)
            continue
        if not node.export:
            continue
        try:
            export_single_node(node, node_result, export_base_path)
        except Exception as e:
            logger.exception("Failed to export node %s: %s", node_id, e)

# ------------------------------------------------------------------ #
# Internal helpers for export
# ------------------------------------------------------------------ #
def export_single_node(
    node: GraphNode,
    result: NodeExecutionResult,
    base_dir: Path,
) -> None:
    # Decide which representation we are exporting & default extension
    if result.structured_output is not None:
        content_kind = "structured"
        default_ext = ".json"
    elif result.tool_call_output is not None:
        content_kind = "tool"
        default_ext = ".json"
    else:
        content_kind = "raw"
        default_ext = ".md"

    # Resolve final path
    filepath = resolve_export_path(
        node=node,
        started_at=result.started_at,
        base_dir=base_dir,
        default_ext=default_ext,
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ext = filepath.suffix.lower() or default_ext

    if ext == ".json":
        payload = build_json_payload(node, result, content_kind)
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    else:
        # markdown / text fallback
        text = build_markdown_payload(node, result, content_kind)
        with filepath.open("w", encoding="utf-8") as f:
            f.write(text)
    logger.info("Exported node %s output to %s", node.id, filepath)

def resolve_export_path(
    node: GraphNode,
    started_at: float,
    base_dir: Path,
    default_ext: str,
) -> Path:
    """
    Decide where to write the file for a node.
    """
    ts = datetime.fromtimestamp(started_at).strftime("%Y%m%d_%H%M%S")
    default_filename = f"{node.id}_{ts}{default_ext}"
    # Node-level override
    if node.export_path:
        p = Path(node.export_path)
        if p.suffix:
            # Treat as explicit file path with its own extension
            return p
        else:
            # Treat as directory/bare path
            return p / default_filename
    # No node.export_path -> use agent-level base_dir
    return base_dir / default_filename

def build_json_payload(
    node: GraphNode,
    result: NodeExecutionResult,
    content_kind: str,
) -> dict:
    """
    Build a JSON-friendly representation of the node result.
    """
    base_meta = {
        "node_id": node.id,
        "mode": getattr(result.mode, "value", str(result.mode)),
        "started_at": result.started_at,
        "duration_ms": result.duration_ms,
        "error": result.error,
    }

    if content_kind == "structured":
        sr: StructuredResponse = result.structured_output  # type: ignore[assignment]
        payload = {
            **base_meta,
            "type": "structured",
            "output_schema": getattr(getattr(sr, "output_schema", None), "__name__", None),
            "model_response": getattr(sr, "model_response", None),
            "json_obj": getattr(sr, "json_obj", None),
        }
        parsed = sr.parsed_output or None
        if parsed is not None:
            # Try to use Pydantic model_dump if available
            try:
                payload["parsed_output"] = parsed.model_dump()
            except Exception:
                payload["parsed_output"] = str(parsed)
        return payload

    if content_kind == "tool":
        tc: ToolCallResponse = result.tool_call_output  # type: ignore[assignment]
        returns_val = getattr(tc, "returns", None)
        from types import GeneratorType

        if isinstance(returns_val, GeneratorType):
            returns_serializable = "<streaming generator; content not captured>"
        else:
            returns_serializable = returns_val

        return {
            **base_meta,
            "type": "tool_call",
            "selected_tool": getattr(tc, "selected_tool", None),
            "inputs": getattr(tc, "inputs", None),
            "returns": returns_serializable,
            "final": getattr(tc, "final", None),
            "extras": getattr(tc, "extras", None),
        }
    # raw
    return {
        **base_meta,
        "type": "raw",
        "raw_output": result.raw_output,
    }

def build_markdown_payload(
    node: GraphNode,
    result: NodeExecutionResult,
    content_kind: str,
) -> str:
    """
    Build a Markdown representation of the node result.
    """
    started = datetime.fromtimestamp(result.started_at).isoformat()
    header = (
        f"# Node `{node.id}` output\n\n"
        f"- Mode: `{getattr(result.mode, 'value', str(result.mode))}`\n"
        f"- Started at: `{started}`\n"
        f"- Duration: `{result.duration_ms}` ms\n"
        f"- Error: `{result.error}`\n\n"
        "---\n\n"
    )
    if content_kind == "structured":
        payload = build_json_payload(node, result, content_kind)
        body = "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n```"
        return header + body

    if content_kind == "tool":
        payload = build_json_payload(node, result, content_kind)
        body = "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n```"
        return header + body

    # raw
    body = str(result.raw_output)
    return header + body