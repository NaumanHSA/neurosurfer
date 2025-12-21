from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, Union

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools import Toolkit
from neurosurfer.tracing import Tracer, TracerConfig
from neurosurfer.agents.rag.agent import RAGAgent
from neurosurfer.agents.rag import RAGAgent, RAGAgentConfig

from .schema import Graph, GraphExecutionResult
from .executor import GraphExecutor   # the class you showed above
from .manager import ManagerConfig
from .artifacts import ArtifactStore
from .loader import load_graph         # or wherever your loader lives
from .export import export


class GraphAgent:
    """
    High-level agent wrapper around `GraphExecutor`.

    Users can think of this as:
      - "Use a normal Agent for single-step reasoning"
      - "Use GraphAgent when you want a DAG/flow of Agents"

    Typical usage
    -------------
        agent = GraphAgent(
            graph_yaml="blog_workflow.yml",
            llm=LLM,
            toolkit=toolkit,
        )

        result = agent.run(
            inputs={
                "topic_title": "...",
                "query": "...",
                "audience": "...",
                "tone": "...",
            }
        )
    """

    def __init__(
        self,
        *,
        id: str = "graph_agent",
        llm: BaseChatModel,
        graph_yaml: Optional[Union[str, Path]] = None,
        graph: Optional[Graph] = None,
        toolkit: Optional[Toolkit] = None,
        manager_llm: Optional[BaseChatModel] = None,
        manager_config: Optional[ManagerConfig] = None,
        tracer: Optional[Tracer] = None,
        artifact_store: Optional[ArtifactStore] = None,
        logger: Optional[logging.Logger] = None,
        log_traces: bool = True,
        export_dir: Union[Path, str] = "exports",
        knowledge_sources: Optional[list[Union[str, Path]]] = None,  # dirs/files for KB
        rag_agent: Optional[RAGAgent] = None,
        auto_ingest_kb: bool = True,
        kb_tool_name: str = "kb_search",
    ) -> None:
        """
        Parameters
        ----------
        llm:
            Main LLM used by node Agents.
        toolkit:
            Global Toolkit. Each node will get a filtered sub-toolkit
            based on its YAML `tools` list.
        graph_yaml:
            Path to a YAML graph definition. If provided, this takes
            precedence over `graph`.
        graph:
            Already-parsed Graph. Used if `graph_yaml` is not given.
        manager_llm:
            LLM for the ManagerAgent. Defaults to `llm` if not provided.
        manager_config:
            Configuration for ManagerAgent. If None, a default ManagerConfig()
            is created.
        tracer:
            Optional Tracer instance used inside GraphExecutor/Agents.
        artifact_store:
            Optional ArtifactStore for node outputs.
        logger:
            Optional logger. If None, a default namespaced logger is used.
        log_traces:
            Whether node-level agents should log traces through the tracer.
        """
        self.id = id
        self.logger = logger or logging.getLogger("neurosurfer.agents.GraphAgent")

        # Resolve graph spec with YAML taking precedence
        if graph_yaml is not None:
            if graph is not None:
                self.logger.warning(
                    "Both `graph_yaml` and `graph` provided; "
                    "using `graph_yaml` and ignoring `graph`."
                )
            graph = load_graph(str(graph_yaml))

        if graph is None:
            raise ValueError("Either `graph_yaml` or `graph` must be provided.")

        if manager_config is None:
            manager_config = ManagerConfig()

        self.graph: Graph = graph
        self.llm = llm
        self.toolkit = toolkit
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "graph_agent",
                "graph": self.graph.model_dump(),
                "model": self.llm.model_name,
                "toolkit": toolkit is not None,
                "log_steps": self.log_traces,
            },
            logger_=logger,
        )
        self.artifacts = artifact_store or ArtifactStore()
        self.log_traces = log_traces
        self.export_dir = export_dir

        # --- RAG wiring ---
        self.rag: Optional[RAGAgent] = rag_agent
        if self.rag is None and knowledge_sources:
            self.rag = RAGAgent(llm=self.llm, config=RAGAgentConfig(), tracer=self.tracer)
        
        if self.rag and knowledge_sources and auto_ingest_kb:
            self.logger.info("Ingesting knowledge base for GraphAgent...")
            self.rag.ingest(sources=knowledge_sources, reset_state=True)

        self.executor = GraphExecutor(
            graph=self.graph,
            llm=self.llm,
            toolkit=self.toolkit,
            rag_agent=self.rag,
            manager_llm=manager_llm or llm,
            manager_config=manager_config,
            tracer=self.tracer,
            artifact_store=self.artifacts,
            logger=self.logger,
            log_traces=self.log_traces,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        inputs: Any,
        *,
        manager_temperature: Optional[float] = None,
        manager_max_new_tokens: Optional[int] = None,
    ) -> GraphExecutionResult:
        """
        Execute the graph once with the given inputs.

        Parameters
        ----------
        inputs:
            Runtime inputs to the graph (dict or scalar). Delegated to
            `normalize_and_validate_graph_inputs` inside GraphExecutor.
        manager_temperature:
            Optional override for ManagerAgent temperature.
        manager_max_new_tokens:
            Optional override for ManagerAgent max_new_tokens.

        Returns
        -------
        GraphExecutionResult
            Contains the graph spec, all node results, and the final outputs.
        """
        with self.tracer(
            agent_id=self.id,
            kind="graph.execute",
            label="agent.graph.execute",
            inputs={
                "inputs": inputs,
                "manager_temperature": manager_temperature,
                "manager_max_new_tokens": manager_max_new_tokens,
            },
        ) as graph_tracer:
            graph_results: GraphExecutionResult = self.executor.run(
                inputs=inputs,
                manager_temperature=manager_temperature,
                manager_max_new_tokens=manager_max_new_tokens,
                trace_step=graph_tracer,
            )
            # export results if configured in the graph nodes.
            export(graph_results=graph_results, export_base_dir=Path(self.export_dir))
            return graph_results

    async def arun(
        self,
        inputs: Any,
        *,
        manager_temperature: Optional[float] = None,
        manager_max_new_tokens: Optional[int] = None,
    ) -> GraphExecutionResult:
        """
        Async wrapper around `run`, useful in async apps / notebooks.

        This simply offloads the sync `run` into a thread executor.
        """
        loop = asyncio.get_running_loop()
        graph_results: GraphExecutionResult = await loop.run_in_executor(
            None,
            lambda: self.run(
                inputs=inputs,
                manager_temperature=manager_temperature,
                manager_max_new_tokens=manager_max_new_tokens,
            ),
        )
        return graph_results

    # Convenience accessors
    def get_artifact(self, node_id: str) -> Any:
        """Shortcut to fetch a node's raw artifact from the ArtifactStore."""
        return self.artifacts.get(node_id)
