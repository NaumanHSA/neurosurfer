from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from neurosurfer.embeddings import Embedder, _LocalEmbedder
from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import CanonicalResponse, GenerationConfig, Message
from neurosurfer.tracing import Tracer, TracerConfig
from neurosurfer.vectorstores import BaseVectorDB, Doc
from neurosurfer.vectorstores.chroma import ChromaVectorStore

from .chunker import Chunker
from .config import (
    ANSWER_BREADTH_MULTIPLIER,
    RETRIEVAL_SCOPE_BASE_K,
    AnswerBreadth,
    RAGAgentConfig,
    RAGIngestorConfig,
    RetrievalMode,
    RetrievalPlan,
    RetrievalScope,
)
from .constants import supported_file_types
from .context_builder import ContextBuilder
from .filereader import FileReader
from .ingestor import RAGIngestor
from .responses import RAGAgentResponse, RetrieveResult
from .templates import (
    RAG_AGENT_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    RETRIEVAL_PLANNER_SYSTEM_PROMPT,
)
from .token_utils import TokenCounter

if TYPE_CHECKING:
    from neurosurfer.agents.oneshot import Agent


@dataclass
class _TrimResult:
    trimmed_context: str
    base_tokens: int
    context_tokens_used: int
    available_for_context: int
    initial_max_new_tokens: int
    final_max_new_tokens: int
    generation_budget: int


def _run_async(coro) -> Any:
    """Run an async coroutine from sync code.

    Uses asyncio.run() when no loop is running; spawns a thread when one is.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _extract_json(text: str, *, return_dict: bool = False) -> Any:
    """Best-effort JSON extraction from LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {} if return_dict else None


class RAGAgent:
    """
    Retrieval core for RAG pipelines. VectorDB- and embedder-agnostic.

    Retrieves documents, trims context to fit the model's token window, and
    delegates generation to the provided ``agent`` (preferred) or ``llm``.
    The RAG agent's job is store / retrieve / prepare prompt — the LLM call
    is owned by the agent passed in.
    """

    def __init__(
        self,
        id: str = "rag_agent",
        llm: Provider | None = None,
        agent: Agent | None = None,
        vectorstore: BaseVectorDB | None = None,
        embedder: Embedder | str | None = None,
        file_reader: FileReader | None = None,
        chunker: Chunker | None = None,
        *,
        config: RAGAgentConfig | None = None,
        ingestor_config: RAGIngestorConfig | None = None,
        make_source=None,
        router_llm: Provider | None = None,
        tracer: Tracer | None = None,
        logger: logging.Logger | None = None,
    ):
        self.id = id
        self.logger = logger or logging.getLogger(__name__)
        self.cfg = config or RAGAgentConfig()
        self.ingestor_cfg = ingestor_config or RAGIngestorConfig()

        # Agent (preferred) or bare provider — RAG delegates generation to one of these.
        self._agent = agent
        if agent is not None and llm is None:
            llm = agent.provider
        self.llm = llm

        # Tracer defaults to silent — enable by passing tracer=Tracer(config=TracerConfig(log_steps=True))
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=False),
            meta={
                "agent_type": self.id,
                "agent_config": self.cfg.__dict__,
                "model": getattr(llm, "model", "unknown"),
            },
            logger_=self.logger,
        )

        self.router_llm = router_llm or llm

        # Per-instance memo of ChromaVectorStore by (collection, clear-on-init) so we
        # don't rebuild a store for a collection we've already opened. Instance-scoped
        # (not a method-level lru_cache) so it never pins the agent in a global cache.
        self._vs_cache: dict[tuple[str, bool], ChromaVectorStore] = {}

        self.vectorstore = vectorstore
        if not self.vectorstore:
            self.logger.warning(
                "No vectorstore provided to RAGAgent, using default ChromaVectorStore. "
                "Initializing default collection `neurosurfer-rag-agent`"
            )
            self.vectorstore = self._vs(self.cfg.collection_name, self.cfg.clear_collection_on_init)

        self.embedder = embedder
        if not self.embedder:
            if not self.cfg.embedding_model:
                raise ValueError("embedder or embedding_model must be provided to RAGAgent config")
            self.logger.warning("No embedder provided to RAGAgent, using default _LocalEmbedder")
            self.logger.info("Initializing Embedding model. This may take a moment...")
            self.embedder = _LocalEmbedder(self.cfg.embedding_model)
        if isinstance(self.embedder, str):
            self.embedder = _LocalEmbedder(self.embedder)

        self.file_reader = file_reader or FileReader()
        self.chunker = chunker or Chunker()

        self.tokens = TokenCounter(self.llm, chars_per_token=self.cfg.approx_chars_per_token)
        self.ctx = ContextBuilder(
            include_metadata_in_context=self.cfg.include_metadata_in_context,
            context_separator=self.cfg.context_separator,
            context_item_header_fmt=self.cfg.context_item_header_fmt,
            make_source=make_source,
        )
        self.ingestor = RAGIngestor(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
            file_reader=self.file_reader,
            chunker=self.chunker,
            logger=self.logger,
            batch_size=self.ingestor_cfg.batch_size,
            max_workers=self.ingestor_cfg.max_workers,
            deduplicate=self.ingestor_cfg.deduplicate,
            normalize_embeddings=self.ingestor_cfg.normalize_embeddings,
            default_metadata=self.ingestor_cfg.default_metadata,
            tmp_dir=self.ingestor_cfg.tmp_dir,
        )

    def _vs(self, collection_name: str, clear_collection_on_init: bool = False) -> ChromaVectorStore:
        key = (collection_name, clear_collection_on_init)
        vs = self._vs_cache.get(key)
        if vs is None:
            vs = ChromaVectorStore(
                collection_name=collection_name,
                clear_collection=clear_collection_on_init,
                persist_directory=self.cfg.persist_directory,
            )
            self._vs_cache[key] = vs
        return vs

    def _collection_has_docs(self) -> bool:
        try:
            return self.vectorstore.count() > 0
        except Exception:
            return False

    def _get_collection_docs_count(self) -> int:
        return self.vectorstore.count()

    # ---------- Public API ----------
    def set_collection(self, collection_name: str, clear_collection_on_init: bool = False) -> None:
        """Set a new collection for ingesting and retrieving documents."""
        self.vectorstore = self._vs(collection_name, clear_collection_on_init)
        self.ingestor.set_vectorstore(self.vectorstore)

    def ingest(
        self,
        sources: str | Path | Iterable[str | Path | str] | None = None,
        *,
        url_fetcher=None,
        include_exts: set[str] | None = supported_file_types,
        extra_metadata: dict[str, Any] | None = None,
        reset_state: bool = True,
    ) -> dict[str, Any]:
        """Unified high-level ingestion wrapper. See RAGIngestor.ingest() for full docs."""
        return self.ingestor.ingest(
            sources=sources,
            url_fetcher=url_fetcher,
            include_exts=include_exts,
            extra_metadata=extra_metadata,
            reset_state=reset_state,
        )

    def retrieve(
        self,
        user_query: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
        retrieval_mode: RetrievalMode = "classic",
        retrieval_scope: RetrievalScope | None = None,
        retrieval_plan: RetrievalPlan | None = None,
        answer_breadth: AnswerBreadth | None = None,
    ) -> RetrieveResult:
        """Retrieve relevant documents from the vectorstore for a given query."""
        self.tracer.reset()

        if not self.vectorstore or not self.embedder:
            raise ValueError("VectorDB and embedder must be provided to RAGAgent")

        if retrieval_mode is not None and retrieval_mode not in get_args(RetrievalMode):
            raise ValueError(f"Retrieval mode must be one of: {get_args(RetrievalMode)}")

        if retrieval_scope is not None and retrieval_scope not in get_args(RetrievalScope):
            raise ValueError(f"Retrieval scope must be one of: {get_args(RetrievalScope)}")

        if retrieval_plan is not None and not isinstance(retrieval_plan, RetrievalPlan):
            raise ValueError("Retrieval plan must be an instance of RetrievalPlan")

        if answer_breadth is not None and answer_breadth not in get_args(AnswerBreadth):
            raise ValueError(f"Answer breadth must be one of: {get_args(AnswerBreadth)}")

        # Decide the plan
        if retrieval_plan is not None:
            plan = retrieval_plan
        elif retrieval_mode == "classic":
            plan = RetrievalPlan(
                mode="classic",
                scope=None,
                answer_breadth=None,
                optimized_query=user_query,
                top_k=top_k or self.cfg.top_k,
            )
        else:
            if retrieval_scope is not None:
                default_k = self._compute_top_k(retrieval_scope, answer_breadth, max_top_k=50)
                plan = RetrievalPlan(
                    mode="smart",
                    scope=retrieval_scope,
                    answer_breadth=answer_breadth,
                    optimized_query=user_query,
                    top_k=default_k,
                )
            else:
                plan = self._plan_retrieval(user_query)

        if not (plan.optimized_query or "").strip():
            self.logger.warning("RAGAgent.retrieve: empty query — returning empty context.")
            return RetrieveResult(
                context="", max_new_tokens=self.cfg.max_new_tokens,
                base_tokens=0, context_tokens_used=0,
                token_budget=self.cfg.max_new_tokens,
                generation_budget=self.cfg.max_new_tokens, docs=[], distances=[],
            )

        docs: list[Any] = []
        distances: list[float] = []
        try:
            query_vec = self.embedder.embed([plan.optimized_query])[0]
            raw = self.vectorstore.similarity_search(
                query_embedding=query_vec,
                top_k=plan.top_k or self.cfg.top_k,
                metadata_filter=metadata_filter,
                similarity_threshold=similarity_threshold or self.cfg.similarity_threshold,
            )
            docs, distances = self._unpack_results(raw)
        except Exception as exc:
            self.logger.error(f"RAGAgent.retrieve: vector search failed: {exc}")
            return RetrieveResult(
                context="", max_new_tokens=self.cfg.max_new_tokens,
                base_tokens=0, context_tokens_used=0,
                token_budget=self.cfg.max_new_tokens,
                generation_budget=self.cfg.max_new_tokens, docs=[], distances=[],
            )

        if not docs:
            self.logger.warning("RAGAgent.retrieve: vector store returned 0 results.")

        untrimmed_context = self.ctx.build(docs)
        trim = self._trim_context_by_token_limit(
            db_context=untrimmed_context,
            reserved_prompt_tokens=getattr(self.cfg, "prompt_token_buffer", 500),
        )

        self.logger.info(
            f"Retrieved {len(docs)} docs — {trim.context_tokens_used:,} context tokens "
            f"(budget: {trim.generation_budget:,} for generation)"
        )

        return RetrieveResult(
            context=trim.trimmed_context,
            max_new_tokens=trim.final_max_new_tokens,
            base_tokens=trim.base_tokens,
            context_tokens_used=trim.context_tokens_used,
            token_budget=self.cfg.max_new_tokens,
            generation_budget=trim.generation_budget,
            docs=docs,
            distances=distances,
            meta={
                "available_for_context": trim.available_for_context,
                "initial_max_new_tokens": trim.initial_max_new_tokens,
                "safety_margin_tokens": self.cfg.safety_margin_tokens,
                "prompt_token_buffer": getattr(self.cfg, "prompt_token_buffer", 500),
            },
        )

    def run(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
        temperature: float | None = None,
        retrieval_mode: RetrievalMode = "classic",
        retrieval_plan: RetrievalPlan | None = None,
        retrieval_scope: RetrievalScope | None = None,
        answer_breadth: AnswerBreadth | None = None,
    ) -> RAGAgentResponse:
        """Retrieve context then generate an answer via the configured agent/provider."""

        if not self.vectorstore or not self.embedder:
            raise ValueError("VectorDB and embedder must be provided to RAGAgent")
        if not self.llm:
            raise ValueError(
                "LLM must be provided to RAGAgent. Pass llm= or agent= at construction."
            )

        retrieved = self.retrieve(
            user_query=user_prompt,
            top_k=top_k or self.cfg.top_k,
            metadata_filter=metadata_filter,
            similarity_threshold=similarity_threshold,
            retrieval_mode=retrieval_mode,
            retrieval_scope=retrieval_scope,
            retrieval_plan=retrieval_plan,
            answer_breadth=answer_breadth,
        )

        sys_prompt = system_prompt or RAG_AGENT_SYSTEM_PROMPT
        augmented_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=retrieved.context, query=user_prompt
        )

        gen_cfg = GenerationConfig(
            max_tokens=retrieved.max_new_tokens or self.cfg.max_new_tokens,
            temperature=temperature or self.cfg.temperature,
        )

        self.logger.info("Generating answer...")

        # Delegate generation: use the agent's provider so the Agent owns the LLM call.
        provider = self._agent.provider if self._agent is not None else self.llm
        agent_response: CanonicalResponse = _run_async(
            provider.complete(
                messages=[Message.user_text(augmented_prompt)],
                system=sys_prompt,
                config=gen_cfg,
            )
        )
        return RAGAgentResponse(rag_retrieval=retrieved, agent_response=agent_response)

    def _plan_retrieval(self, user_query: str) -> RetrievalPlan:
        """Use router_llm to decide retrieval scope and top_k when in 'smart' mode."""
        if self.router_llm is None:
            return RetrievalPlan(
                mode="smart",
                scope="medium",
                answer_breadth="single_fact",
                top_k=self.cfg.top_k,
            )
        user_prompt = (
            "You will receive a USER PROMPT between the markers below.\n"
            "\n"
            "===BEGIN_USER_PROMPT===\n"
            f"{user_query}\n"
            "===END_USER_PROMPT===\n"
            "\n"
            "Task:"
            "1) Decide the retrieval scope and answer breadth needed to answer the USER PROMPT."
            "2) Produce an optimized_query suitable for embedding-based retrieval."
            "\n"
            "Output:"
            "Return ONLY a single JSON object with keys: scope, answer_breadth, optimized_query."
        )

        try:
            response = _run_async(
                self.router_llm.complete(
                    messages=[Message.user_text(user_prompt)],
                    system=RETRIEVAL_PLANNER_SYSTEM_PROMPT,
                    config=GenerationConfig(),
                )
            )
            content = response.text()

            obj = _extract_json(content, return_dict=True)
            scope = obj.get("scope", "medium")
            answer_breadth = obj.get("answer_breadth", "single_fact")
            optimized_query = obj.get("optimized_query", "")

            top_k = self._compute_top_k(scope, answer_breadth)
            return RetrievalPlan(
                mode="smart",
                scope=scope,
                answer_breadth=answer_breadth,
                top_k=top_k,
                optimized_query=optimized_query,
            )
        except Exception:
            self.logger.exception("RAGAgent._plan_retrieval failed; using default plan.")
            return RetrievalPlan(
                mode="smart",
                scope="medium",
                answer_breadth="single_fact",
                optimized_query=user_query,
                top_k=self.cfg.top_k,
            )

    def _compute_top_k(self, scope: str, breadth: str, max_top_k: int = 50) -> int:
        base_k = RETRIEVAL_SCOPE_BASE_K.get(scope, 10)
        mult = ANSWER_BREADTH_MULTIPLIER.get(breadth, 1)
        return min(base_k * mult, max_top_k)

    # ---------- Internals ----------
    def _trim_context_by_token_limit(
        self,
        db_context: str,
        *,
        reserved_prompt_tokens: int | None = None,
    ) -> _TrimResult:
        """Trim DB context to fit within the model's token budget."""
        provider_ctx = getattr(getattr(self.llm, "capabilities", None), "context_window", None)
        max_seq = provider_ctx or self.cfg.max_new_tokens
        max_ctx = max_seq - int(self.cfg.safety_margin_tokens)

        prompt_buffer = int(reserved_prompt_tokens or self.cfg.prompt_token_buffer)
        min_output_tokens = int(self.cfg.min_output_tokens)
        if prompt_buffer + min_output_tokens > max_ctx:
            prompt_buffer = max(0, max_ctx - min_output_tokens)
        base_tokens = prompt_buffer
        if self.cfg.fixed_max_new_tokens is not None:
            initial_max_new_tokens = int(self.cfg.fixed_max_new_tokens)
        else:
            remaining = max(max_ctx - base_tokens, 0)
            initial_max_new_tokens = max(
                int(remaining * self.cfg.auto_output_ratio),
                min_output_tokens,
            )
        available_for_context = max(max_ctx - base_tokens - initial_max_new_tokens, 0)
        trimmed_context, context_tokens_used = self.tokens.trim_to_tokens(
            db_context, available_for_context
        )
        final_max_new_tokens = self._calculate_final_max_new_tokens(
            fixed_max_new_tokens=self.cfg.fixed_max_new_tokens,
            min_output_tokens=min_output_tokens,
            base_tokens=base_tokens,
            context_tokens_used=context_tokens_used,
            max_ctx=max_ctx,
        )
        generation_budget = max(max_ctx - base_tokens - context_tokens_used, 0)
        return _TrimResult(
            trimmed_context=trimmed_context,
            base_tokens=base_tokens,
            context_tokens_used=context_tokens_used,
            available_for_context=available_for_context,
            initial_max_new_tokens=initial_max_new_tokens,
            final_max_new_tokens=final_max_new_tokens,
            generation_budget=generation_budget,
        )

    @staticmethod
    def _calculate_final_max_new_tokens(
        fixed_max_new_tokens: int | None,
        min_output_tokens: int,
        base_tokens: int,
        context_tokens_used: int,
        max_ctx: int,
    ) -> int:
        remaining = max(max_ctx - base_tokens - context_tokens_used, 0)
        if fixed_max_new_tokens is not None:
            return max(min(fixed_max_new_tokens, remaining), 0)
        return max(remaining, min_output_tokens)

    @staticmethod
    def _unpack_results(
        raw: list[Doc] | list[tuple[Doc, float]]
    ) -> tuple[list[Doc], list[float | None]]:
        docs: list[Doc] = []
        dists: list[float | None] = []
        if not raw:
            return docs, dists
        first = raw[0]
        if isinstance(first, tuple):
            for d, dist in raw:  # type: ignore[misc]
                docs.append(d)
                dists.append(dist)
        else:
            docs = raw  # type: ignore[assignment]
            dists = [None] * len(docs)
        return docs, dists
