from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from neurosurfer.agents.rag import RAGAgent, RAGAgentConfig, RAGIngestorConfig


logger = logging.getLogger(__name__)


class KnowledgeBaseTool(BaseTool):
    """
    Tool that exposes a shared RAG-powered knowledge base to agents/graph nodes.

    Typical usage
    -------------
    1. Create or reuse a `RAGAgent` for the project:

        rag = RAGAgent(llm=LLM, config=RAGAgentConfig())

    2. Optionally ingest code/docs:

        rag.ingest(sources=["./neurosurfer", "./docs"], reset_state=True)

    3. Wrap it in a KnowledgeBaseTool and register in a Toolkit:

        kb_tool = KnowledgeBaseTool(
            rag=rag,
            config=KnowledgeBaseToolConfig(
                name="kb_search",
                knowledge_sources=None,  # already ingested above
            )
        )
        toolkit.register_tool(kb_tool)

    4. In graphs or plain agents, any node/agent that has this tool
       can query the KB with natural language.

    Behaviour
    ---------
    - `__init__`:
        * Optionally performs ingestion if `config.knowledge_sources` and
          `config.auto_ingest` are set.
    - `__call__`:
        * Accepts a query and optional retrieval controls (top_k, scope, metadata filter).
        * Calls `RAGAgent.retrieve(...)` and returns a JSON-like payload
          containing the retrieved context, documents, and metadata.
    """

    spec = ToolSpec(
        name="kb_search",  # will be overridden by config.name
        description=(
            "Query the shared knowledge base (code + docs + other ingested content) "
            "using a RAG pipeline. Returns trimmed context plus document metadata "
            "for use in reasoning and generation."
        ),
        when_to_use=(
            "Use this tool when the user query requires information from the "
            "project's codebase, documentation, or any ingested domain files. "
            "Ideal for doc generation, code explanation, and architecture Q&A."
        ),
        inputs=[
            ToolParam(
                name="query",
                type="string",
                description="Natural language query against the knowledge base.",
                required=True,
                llm=True,
            ),
            ToolParam(
                name="top_k",
                type="integer",
                description="Optional number of chunks to retrieve (overrides default).",
                required=False,
                llm=True,
            ),
            ToolParam(
                name="retrieval_scope",
                type="string",
                description="Optional retrieval scope (small | medium | wide | full). If omitted, the RAGAgent decides based on its config or planner.",
                required=False,
                llm=True,
            ),
            ToolParam(
                name="metadata_filter",
                type="object",
                description=(
                    "Optional metadata filter object passed to the underlying vector store. "
                    "For example: {\"file_path\": \"neurosurfer/agents/graph/executor.py\"}."
                ),
                required=False,
                llm=False,  # usually wired programmatically, not by LLM
            ),
        ],
        returns=ToolReturn(
            type="object",
            description=(
                "A JSON object with keys:\n"
                "- query: original query string\n"
                "- context: trimmed concatenated context string for direct use in prompts\n"
                "- docs: list of {id, content, metadata, distance}\n"
                "- meta: retrieval metadata (token budgets, etc.)"
            ),
        ),
    )

    def __init__(
        self,
        *,
        rag: RAGAgent,
        knowledge_sources: Optional[Union[str, Path, Iterable[Union[str, Path, str]]]] = None,
        rag_config: Optional[RAGAgentConfig] = None,
        ingestor_config: Optional[RAGIngestorConfig] = None,
        auto_ingest: bool = True,
    ) -> None:
        """
        Initialize the KnowledgeBaseTool.

        Parameters
        ----------
        rag:
            A pre-initialized RAGAgent instance. It will be used for both
            ingestion (if any) and retrieval.
        name:
            Tool name to register in the Toolkit (e.g. "kb_search").
        knowledge_sources:
            Optional sources to ingest into the RAGAgent on initialization.
            This is passed directly to `RAGAgent.ingest`, so it can be:
              - a single file/directory path (str or Path)
              - a list/iterable of paths/URLs/texts
        rag_config:
            Optional RAGAgentConfig. If not provided, a default is created.
        ingestor_config:
            Optional RAGIngestorConfig. If not provided, a default is created.
        auto_ingest:
            If True and knowledge_sources is provided, ingest at init time.
            (not used here, but kept for future extension).
        """
        super().__init__()
        self.rag = rag
        self.knowledge_sources = knowledge_sources
        self.rag_config = rag_config
        self.ingestor_config = ingestor_config
        self.auto_ingest = auto_ingest

        # Optionally run ingestion at initialization.
        if self.auto_ingest and self.knowledge_sources:
            try:
                logger.info(
                    "[KnowledgeBaseTool] Ingesting knowledge sources into RAGAgent: %s",
                    self.knowledge_sources,
                )
                summary = self.rag.ingest(
                    sources=self.knowledge_sources,
                    reset_state=True,
                )
                status = summary.get("status", "ok") if isinstance(summary, dict) else "ok"
                logger.info(
                    "[KnowledgeBaseTool] Ingestion finished with status=%s; summary=%s",
                    status,
                    summary,
                )
            except Exception:
                logger.exception(
                    "[KnowledgeBaseTool] Failed to ingest knowledge sources: %s",
                    self.knowledge_sources,
                )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieval_scope: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> ToolResponse:
        """
        Query the shared knowledge base via the underlying RAGAgent.

        Parameters
        ----------
        query:
            Natural language query about the codebase / docs / ingested content.
        top_k:
            Optional number of chunks to retrieve. If None, the RAGAgent's
            config (or planner) determines the value.
        retrieval_scope:
            Optional retrieval scope string. Must match the RAGAgent's
            RetrievalScope enum ("small", "medium", "wide", "full"), if used.
        metadata_filter:
            Optional metadata filter dictionary for the vector store.

        Returns
        -------
        ToolResponse
            ToolResponse.results will contain an object:
                {
                    "query": str,
                    "context": str,
                    "docs": [
                        {
                            "id": ...,
                            "content": ...,
                            "metadata": ...,
                            "distance": ...,
                        },
                        ...
                    ],
                    "meta": {...}
                }
            ToolResponse.final_answer is False (this tool does not finalize).
        """
        try:
            retrieve_result = self.rag.retrieve(
                user_query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                retrieval_scope=retrieval_scope,  # RAGAgent enforces valid values
            )
        except Exception as e:
            logger.exception("[KnowledgeBaseTool] RAGAgent.retrieve failed: %s", e)
            return ToolResponse(
                results={
                    "query": query,
                    "error": f"Knowledge base retrieval failed: {e}",
                    "context": "",
                    "docs": [],
                    "meta": {},
                },
                final_answer=False,
                extras={},
            )

        docs_payload: List[Dict[str, Any]] = []
        for doc, dist in zip(retrieve_result.docs, retrieve_result.distances):
            docs_payload.append(
                {
                    "id": getattr(doc, "id", None),
                    "content": getattr(doc, "page_content", ""),
                    "metadata": getattr(doc, "metadata", {}) or {},
                    "distance": dist,
                }
            )

        results_dict: Dict[str, Any] = {
            "query": query,
            "context": retrieve_result.context,
            "docs": docs_payload,
            "meta": retrieve_result.meta,
        }

        return ToolResponse(
            results=results_dict,
            final_answer=False,
            extras={},
        )
