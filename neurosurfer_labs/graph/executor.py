from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel as PydModel

from neurosurfer.models.chat_models.base import BaseModel as ChatBaseModel
from neurosurfer.tools import Toolkit

from .artifacts import ArtifactStore
from .errors import GraphConfigurationError
from .manager import ManagerAgent
from .schema import GraphSpec, GraphNode
from .templates import DEFAULT_NODE_SYSTEM_TEMPLATE
from .types import NodeExecutionResult, NodeMode
from .utils import topo_sort, import_string
from neurosurfer.agents.common.tracing import Tracer, NullTracer
from neurosurfer.agents.agent.responses import StructuredResponse, ToolCallResponse
from neurosurfer.agents import Agent, AgentConfig


class GraphExecutor:
    """
    Execute a `GraphSpec` DAG using:
      - A single shared `llm` (ChatBaseModel) for all nodes by default.
      - A single shared `Toolkit` (all tools), with per-node subsets based on YAML.
      - A `ManagerAgent` (by default using the same `llm`) to compose inter-node prompts.

    Users only need:
      - YAML flow (GraphSpec)
      - `llm` instance
      - `toolkit` instance

    Everything else (Agents, tool subsets, manager, etc.) is wired automatically.
    """

    def __init__(
        self,
        graph: GraphSpec,
        *,
        llm: ChatBaseModel,
        toolkit: Optional[Toolkit] = None,
        manager_llm: Optional[ChatBaseModel] = None,
        agent_config: Optional[AgentConfig] = None,
        tracer: Optional[Tracer] = None,
        artifact_store: Optional[ArtifactStore] = None,
        logger: Optional[logging.Logger] = None,
        enable_tracing: Optional[bool] = None,
    ) -> None:
        self.graph = graph
        self.llm = llm
        self.toolkit = toolkit
        self.agent_config = agent_config or AgentConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.artifacts = artifact_store or ArtifactStore()

        # Manager uses same LLM by default
        self.manager = ManagerAgent(manager_llm or llm)

        # Tracing
        self._base_tracer: Tracer = tracer or NullTracer()
        self._null_tracer: Tracer = NullTracer()
        self._enable_tracing_default: bool = (
            bool(enable_tracing) if enable_tracing is not None else True
        )

        self._node_map: Dict[str, GraphNode] = self.graph.node_map()
        self._order = topo_sort(self.graph.nodes)

        # Lazy-created Agents per node
        self._agents: Dict[str, Agent] = {}

        # Validate tools early if toolkit is provided
        if self.toolkit is not None:
            self._validate_tools()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        inputs: Dict[str, Any],
        *,
        trace: Optional[bool] = None,
        manager_temperature: float = 0.2,
        manager_max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Execute the entire graph once.

        Parameters
        ----------
        inputs:
            Arbitrary JSON-serializable dict representing the graph input.
        trace:
            Optional per-call override to enable/disable tracing.
        manager_temperature:
            Temperature used for ManagerAgent when composing prompts.
        manager_max_new_tokens:
            Max new tokens for ManagerAgent responses.

        Returns
        -------
        Dict[str, Any]
            {
              "graph": GraphSpec,
              "results": Dict[node_id, NodeExecutionResult],
              "final": Dict[node_id, Any],  # raw_output for final nodes
            }
        """
        tracer = self._get_tracer(trace)

        results: Dict[str, NodeExecutionResult] = {}
        last_result: Optional[NodeExecutionResult] = None

        with tracer.span(
            "graph.run",
            {
                "graph_name": self.graph.name,
                "num_nodes": len(self.graph.nodes),
            },
        ):
            for node_id in self._order:
                node = self._node_map[node_id]
                dep_results = {
                    dep_id: results[dep_id].raw_output for dep_id in node.depends_on
                }
                prev = last_result.raw_output if last_result else None

                node_result = self._run_node(
                    node=node,
                    graph_inputs=inputs,
                    dependency_results=dep_results,
                    previous_result=prev,
                    tracer=tracer,
                    manager_temperature=manager_temperature,
                    manager_max_new_tokens=manager_max_new_tokens,
                )
                results[node_id] = node_result
                last_result = node_result

        final = self._select_final_outputs(results)
        return {
            "graph": self.graph,
            "results": results,
            "final": final,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_tracer(self, trace: Optional[bool]) -> Tracer:
        effective = self._enable_tracing_default if trace is None else bool(trace)
        return self._base_tracer if effective else self._null_tracer

    def _validate_tools(self) -> None:
        """Ensure all YAML tool names exist in the master Toolkit."""
        if self.toolkit is None:
            return

        missing: set[str] = set()
        for node in self.graph.nodes:
            for t_name in node.tools:
                if t_name not in self.toolkit.registry:
                    missing.add(t_name)
        if missing:
            raise GraphConfigurationError(
                f"YAML refers to unknown tools not registered in Toolkit: {sorted(missing)}"
            )

    def _get_agent_for_node(self, node: GraphNode) -> Agent:
        """Create or reuse an Agent instance for this node."""
        if node.id in self._agents:
            return self._agents[node.id]

        # Per-node toolkit: restrict to YAML tools (if any)
        node_toolkit: Optional[Toolkit] = None
        if self.toolkit is not None and node.tools:
            from neurosurfer.tools.toolkit import Toolkit as ToolkitClass

            node_toolkit = ToolkitClass()
            for name in node.tools:
                tool = self.toolkit.registry.get(name)
                if tool is None:
                    raise GraphConfigurationError(
                        f"Node {node.id!r} refers to unknown tool {name!r}"
                    )
                node_toolkit.register_tool(tool)

        agent_logger = logging.getLogger(f"neurosurfer.agent.{node.id}")

        agent = Agent(
            llm=self.llm,
            toolkit=node_toolkit,
            config=self.agent_config,
            logger=agent_logger,
            verbose=True,
        )

        self._agents[node.id] = agent
        return agent

    def _run_node(
        self,
        *,
        node: GraphNode,
        graph_inputs: Dict[str, Any],
        dependency_results: Dict[str, Any],
        previous_result: Any,
        tracer: Tracer,
        manager_temperature: float,
        manager_max_new_tokens: int,
    ) -> NodeExecutionResult:
        agent = self._get_agent_for_node(node)

        with tracer.span(
            "graph.node.start",
            {
                "node_id": node.id,
                "mode": node.mode.value,
                "depends_on": node.depends_on,
                "tools": node.tools,
            },
        ):
            user_prompt = self.manager.compose_user_prompt(
                node=node,
                graph_inputs=graph_inputs,
                dependency_results=dependency_results,
                previous_result=previous_result,
                temperature=manager_temperature,
                max_new_tokens=manager_max_new_tokens,
            )

        system_prompt = self._build_system_prompt(node)
        output_schema = self._load_output_schema_if_needed(node)

        started_at = time.time()

        with tracer.span(
            "graph.node.agent_run",
            {
                "node_id": node.id,
                "mode": node.mode.value,
                "has_schema": bool(output_schema),
            },
        ):
            try:
                run_kwargs: Dict[str, Any] = {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "temperature": None,       # let AgentConfig decide default
                    "max_new_tokens": None,    # let AgentConfig decide default
                    "stream": False,
                    "context": {
                        "graph_inputs": graph_inputs,
                        "dependencies": dependency_results,
                    },
                }

                if output_schema is not None:
                    run_kwargs["output_schema"] = output_schema

                if node.strict_tool_call is not None:
                    run_kwargs["strict_tool_call"] = node.strict_tool_call

                result = agent.run(**run_kwargs)

                raw, structured, tool_call = self._normalize_agent_output(result)
                duration_ms = int((time.time() - started_at) * 1000)

                ner = NodeExecutionResult(
                    node_id=node.id,
                    mode=node.mode,
                    raw_output=raw,
                    structured_output=structured,
                    tool_call_output=tool_call,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    error=None,
                )
                self.artifacts.put(node.id, raw)
                return ner

            except Exception as e:
                duration_ms = int((time.time() - started_at) * 1000)
                self.logger.exception("Node %s failed: %s", node.id, e)
                return NodeExecutionResult(
                    node_id=node.id,
                    mode=node.mode,
                    raw_output=None,
                    structured_output=None,
                    tool_call_output=None,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    error=str(e),
                )

    def _build_system_prompt(self, node: GraphNode) -> str:
        purpose = node.purpose or node.description or f"Node {node.id}"
        goal = node.goal or "Follow the instructions in the user prompt."
        expected = node.expected_result or "A useful, correct, and concise answer."

        return DEFAULT_NODE_SYSTEM_TEMPLATE.format(
            purpose=purpose,
            goal=goal,
            expected_result=expected,
        )

    def _load_output_schema_if_needed(
        self,
        node: GraphNode,
    ) -> Optional[type[PydModel]]:
        if not node.output_schema:
            return None

        obj = import_string(node.output_schema)
        if not isinstance(obj, type) or not issubclass(obj, PydModel):
            raise GraphConfigurationError(
                f"output_schema {node.output_schema!r} is not a Pydantic model"
            )
        return obj

    def _normalize_agent_output(
        self, result: Any
    ) -> Tuple[Any, Optional[StructuredResponse], Optional[ToolCallResponse]]:
        """
        Collapse the variety of Agent.run outputs into a single representation.
        """
        structured: Optional[StructuredResponse] = None
        tool_call: Optional[ToolCallResponse] = None
        raw: Any = result

        if isinstance(result, StructuredResponse):
            structured = result
            if result.parsed_output is not None:
                raw = result.parsed_output
            elif result.json_obj is not None:
                raw = result.json_obj
            else:
                raw = result.model_response
        elif isinstance(result, ToolCallResponse):
            tool_call = result
            raw = result.returns

        return raw, structured, tool_call

    def _select_final_outputs(
        self, results: Dict[str, NodeExecutionResult]
    ) -> Dict[str, Any]:
        """
        Pick which node outputs are considered "final" for the graph.

        If `graph.outputs` is empty, use the last node in the topological order.
        """
        if self.graph.outputs:
            return {
                nid: results[nid].raw_output
                for nid in self.graph.outputs
                if nid in results
            }

        if not self._order:
            return {}
        last_nid = self._order[-1]
        if last_nid not in results:
            return {}
        return {last_nid: results[last_nid].raw_output}
