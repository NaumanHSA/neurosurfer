from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import GeneratorType
from typing import Optional, Union, Dict, Any

from neurosurfer.tracing import Tracer, TracerConfig, TraceStepContext
from neurosurfer.agents.agent.responses import StructuredResponse, ToolCallResponse
from .schema import GraphExecutionResult, GraphNode, NodeExecutionResult

LOGGER = logging.getLogger(__name__)


@dataclass
class GraphExportConfig:
    """
    Configuration for exporting graph node results.

    Notes
    -----
    - export_base_dir is the default target directory when node.export_path is missing.
    - include_trace_dump: if True and traces exist, exporter writes traces JSON next to node outputs.
    - traces_filename: if set, writes a single file with the whole graph traces dump.
    """
    export_base_dir: Union[str, Path] = "exports"
    include_trace_dump: bool = False
    traces_filename: Optional[str] = None  # e.g. "traces_{ts}.json"
    default_raw_ext: str = ".md"
    default_json_ext: str = ".json"


class GraphExporter:
    """
    Export graph node results to disk for nodes that have `export=True`.

    Rules
    -----
    - If node.export_path is a file (has extension) -> use that exact file.
    - If node.export_path is a directory / bare path -> write
      `{node_id}_{timestamp}.ext` inside it.
    - If node.export_path is not set -> write into export_base_dir as
      `{node_id}_{timestamp}.ext`.

    Content selection
    -----------------
    - If structured_output is present -> JSON by default.
    - Else if tool_call_output is present -> JSON by default.
    - Else -> raw_output as Markdown by default.

    Format override by extension
    ----------------------------
    - If final extension is `.json`, always write JSON.
    - Otherwise write Markdown/text (embedding JSON if structured/tool output).
    """

    def __init__(
        self,
        *,
        config: Optional[GraphExportConfig] = None,
        tracer: Optional[Tracer] = None,
        logger: Optional[logging.Logger] = None,
        log_traces: bool = True,
    ) -> None:
        self.config = config or GraphExportConfig()
        self.logger = logger or LOGGER
        self.log_traces = log_traces

        self.tracer: Optional[Tracer] = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "GraphExporter",
                "agent_config": self.config,
                "log_steps": self.log_traces,
            },
            logger_=self.logger,
        )

    def export(
        self,
        graph_results: GraphExecutionResult,
    ) -> None:
        """
        Export all nodes with `export=True`.
        """
        graph = graph_results.graph
        node_map = graph.node_map()

        with self.tracer(
            agent_id="graph_exporter",
            kind="export",
            label="graph_exporter.export",
            inputs={
                "nodes_total": len(graph_results.nodes),
                "include_trace_dump": self.config.include_trace_dump,
                "traces_filename": self.config.traces_filename,
            },
        ) as trace_step:

            self._resolve_base_dir(trace_step)
            trace_step.inputs(export_dir=str(self.config.export_base_dir))
            try:
                for node_id, node_result in graph_results.nodes.items():
                    node: Optional[GraphNode] = node_map.get(node_id)
                    if not node:
                        trace_step.log(message=f"No GraphNode found for node_id={node_id}; skipping export.", type="warning")
                        continue
                    if not getattr(node, "export", False):
                        continue
                    try:
                        filepath = self.export_single_node(node=node, result=node_result)
                        trace_step.log(message=f"Exported node {node.id} output to {filepath}", type="info")        
                    except Exception as e:
                        trace_step.log(message=f"Failed to export node {node_id}: {e}", type="error")

                # Optional: write traces (global)
                out = self.export_traces(graph_results=graph_results)
                trace_step.log(message=f"Exported graph traces to {out}", type="info")
            finally:
                trace_step.outputs(exported_nodes_total=len(graph_results.nodes))

    def export_single_node(
        self,
        *,
        node: GraphNode,
        result: NodeExecutionResult,
    ) -> Path:
        """
        Export a single node. Returns the written path.
        """
        content_kind, default_ext = self._decide_kind_and_default_ext(result)
        filepath = self._resolve_export_path(
            node=node,
            started_at=result.started_at,
            default_ext=default_ext,
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)

        ext = (filepath.suffix.lower() or default_ext).lower()

        if ext == ".json":
            payload = self._build_json_payload(node=node, result=result, content_kind=content_kind)
            filepath.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        else:
            text = self._build_markdown_payload(node=node, result=result, content_kind=content_kind)
            filepath.write_text(text, encoding="utf-8")
        return filepath

    def _resolve_base_dir(self, trace_step: Optional[TraceStepContext] = None) -> None:
        base = Path(self.config.export_base_dir)
        if base.exists() and not base.is_dir():
            trace_step.log(message=f"Export base path exists but is not a directory: {base}. Falling back to 'exports'.", type="warning")
            base = Path("exports")
        base.mkdir(parents=True, exist_ok=True)
        self.config.export_base_dir = base

    def _decide_kind_and_default_ext(self, result: NodeExecutionResult) -> tuple[str, str]:
        if result.structured_output is not None:
            return "structured", self.config.default_json_ext
        if result.tool_call_output is not None:
            return "tool", self.config.default_json_ext
        return "raw", self.config.default_raw_ext

    def _resolve_export_path(
        self,
        *,
        node: GraphNode,
        started_at: float,
        default_ext: str,
    ) -> Path:
        ts = datetime.fromtimestamp(started_at).strftime("%Y%m%d_%H%M%S")
        default_filename = f"{node.id}_{ts}{default_ext}"

        export_path = getattr(node, "export_path", None)
        if export_path:
            p = Path(export_path)
            if p.suffix:
                return p
            return p / default_filename

        return self.config.export_base_dir / default_filename

    def _base_meta(self, node: GraphNode, result: NodeExecutionResult) -> Dict[str, Any]:
        return {
            "node_id": node.id,
            "mode": getattr(result.mode, "value", str(result.mode)),
            "started_at": result.started_at,
            "duration_ms": result.duration_ms,
            "error": result.error,
        }

    def _build_json_payload(
        self,
        *,
        node: GraphNode,
        result: NodeExecutionResult,
        content_kind: str,
    ) -> Dict[str, Any]:
        base_meta = self._base_meta(node, result)

        if content_kind == "structured":
            sr: StructuredResponse = result.structured_output  # type: ignore[assignment]
            payload: Dict[str, Any] = {
                **base_meta,
                "type": "structured",
                "output_schema": getattr(getattr(sr, "output_schema", None), "__name__", None),
                "model_response": getattr(sr, "model_response", None),
                "json_obj": getattr(sr, "json_obj", None),
            }

            parsed = getattr(sr, "parsed_output", None) or None
            if parsed is not None:
                try:
                    payload["parsed_output"] = parsed.model_dump()
                except Exception:
                    payload["parsed_output"] = str(parsed)

            return payload

        if content_kind == "tool":
            tc: ToolCallResponse = result.tool_call_output  # type: ignore[assignment]
            returns_val = getattr(tc, "returns", None)

            if isinstance(returns_val, GeneratorType):
                # avoid consuming generators here (side-effects)
                returns_serializable = "<streaming generator; not captured by exporter>"
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

    def _build_markdown_payload(
        self,
        *,
        node: GraphNode,
        result: NodeExecutionResult,
        content_kind: str,
    ) -> str:
        started = datetime.fromtimestamp(result.started_at).isoformat()
        header = (
            f"# Node `{node.id}` output\n\n"
            f"- Mode: `{getattr(result.mode, 'value', str(result.mode))}`\n"
            f"- Started at: `{started}`\n"
            f"- Duration: `{result.duration_ms}` ms\n"
            f"- Error: `{result.error}`\n\n"
            "---\n\n"
        )

        if content_kind in {"structured", "tool"}:
            payload = self._build_json_payload(node=node, result=result, content_kind=content_kind)
            return header + "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n```"

        return header + ("" if result.raw_output is None else str(result.raw_output))

    def export_traces(self, *, graph_results: GraphExecutionResult) -> Path:
        if not self.config.include_trace_dump:
            return

        # 1) per-node traces next to their exports would require knowing file paths
        #    We'll do a single graph-level trace file, optionally.
        if not self.config.traces_filename:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.config.traces_filename.format(ts=ts)
        out = self.config.export_base_dir / filename

        # collect node traces
        traces: Dict[str, Any] = {}
        for node_id, node_res in graph_results.nodes.items():
            t = getattr(node_res, "traces", None)
            if t is None:
                continue
            try:
                traces[node_id] = t.model_dump()
            except Exception:
                traces[node_id] = str(t)
        out.write_text(json.dumps({
            "type": "graph_traces", 
            "nodes": traces
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return out
