from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Callable, Generator, get_args
import logging
from functools import lru_cache

from neurosurfer.vectorstores import Doc, BaseVectorDB
from neurosurfer.models.chat_models.base import BaseChatModel, LLM_RESPONSE_TYPE
from neurosurfer.models.embedders import BaseEmbedder
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.vectorstores.chroma import ChromaVectorStore
from neurosurfer.agents.common.utils import extract_and_repair_json

from .config import (
    RAGAgentConfig, 
    RetrieveResult, 
    RAGIngestorConfig, 
    RetrievalPlan, 
    RetrievalScope, 
    RetrievalMode, 
    AnswerBreadth,
    RETRIEVAL_SCOPE_BASE_K,
    ANSWER_BREADTH_MULTIPLIER,
)
from .token_utils import TokenCounter
from .context_builder import ContextBuilder
from .filereader import FileReader
from .chunker import Chunker
from .ingestor import RAGIngestor
from .constants import supported_file_types
from .templates import RAG_AGENT_SYSTEM_PROMPT, RAG_USER_PROMPT, RETRIEVAL_PLANNER_SYSTEM_PROMPT


@dataclass
class _TrimResult:
    trimmed_context: str
    base_tokens: int
    context_tokens_used: int
    available_for_context: int
    initial_max_new_tokens: int
    final_max_new_tokens: int
    generation_budget: int


class RAGAgent:
    """
    Retrieval core for RAG pipelines. VectorDB- and embedder-agnostic.
    Adds a convenient `run(...)` that makes a full LLM call with the retrieved context.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        vectorstore: Optional[BaseVectorDB] = None,
        embedder: Optional[Union[BaseEmbedder, str]] = None,
        file_reader: Optional[FileReader] = None,
        chunker: Optional[Chunker] = None,
        *,
        config: Optional[RAGAgentConfig] = None,
        ingestor_config: Optional[RAGIngestorConfig] = None,
        logger: Optional[logging.Logger] = None,
        make_source=None,
        router_llm: Optional[BaseChatModel] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.llm = llm
        self.router_llm = router_llm or llm
        self.cfg = config or RAGAgentConfig()
        self.ingestor_cfg = ingestor_config or RAGIngestorConfig()

        self.vectorstore = vectorstore
        if not self.vectorstore:
            self.logger.warning("No vectorstore provided to RAGAgent, using default ChromaVectorStore. Initializing default collection `neurosurfer-rag-agent`")
            self.vectorstore = self._vs(self.cfg.collection_name, self.cfg.clear_collection_on_init)

        self.embedder = embedder
        if not self.embedder:
            if not self.cfg.embedding_model:
                raise ValueError("embedder or embedding_model must be provided to RAGAgent config")
            self.logger.warning("No embedder provided to RAGAgent, using default SentenceTransformerEmbedder")
            self.logger.info("Initializing Embedding model. This may take a moment...")
            self.embedder = SentenceTransformerEmbedder(self.cfg.embedding_model)
        if isinstance(self.embedder, str):
            self.embedder = SentenceTransformerEmbedder(self.embedder)

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

    @lru_cache(maxsize=512)
    def _vs(self, collection_name: str, clear_collection_on_init: bool = False) -> ChromaVectorStore:
        """
        Get or create vector store for collection (cached).
        Uses LRU cache to reuse vector store instances for performance.
        Args:
            collection_name (str): Collection name
            clear_collection_on_init (bool): Whether to clear the collection on initialization
        Returns:
            ChromaVectorStore: Vector store instance
        """
        return ChromaVectorStore(
            collection_name=collection_name,
            clear_collection=clear_collection_on_init,
            persist_directory=self.cfg.persist_directory,
        )

    def _collection_has_docs(self) -> bool:
        """Check if collection has any documents. Returns True if collection has documents, False otherwise"""
        try:
            return self.vectorstore.count() > 0
        except Exception:
            return False
    
    def _get_collection_docs_count(self) -> int:
        """Get all documents in the collection"""
        return self.vectorstore.count()

    # ---------- Public API ----------
    def set_collection(self, collection_name: str, clear_collection_on_init: bool = False) -> None:
        """Call this method to set a new collection for ingesting and retrieving documents."""
        self.vectorstore = self._vs(collection_name, clear_collection_on_init)
        self.ingestor.set_vectorstore(self.vectorstore)

    def ingest(
        self,
        sources: Optional[Union[str, Path, Iterable[Union[str, Path, str]]]] = None,
        *,
        url_fetcher: Optional[Callable[[str], Optional[str]]] = None,
        include_exts: Optional[set[str]] = supported_file_types,
        extra_metadata: Optional[Dict[str, Any]] = None,
        reset_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Unified high-level ingestion wrapper.

        This method accepts a mix of paths, URLs, and raw text, routes them to the
        appropriate add_* methods, and then runs the full pipeline via `build()`.

        Supported source forms:
            - File path (string or Path)
            - Directory path (string or Path)
            - .zip archive (string or Path)
            - Git repo folder (directory containing a .git subfolder)
            - URL string (starting with http:// or https://)
            - Raw text string (everything else)

        Parameters
        ----------
        sources:
            A single source or iterable of sources. Each element can be:
            - str
            - pathlib.Path
        url_fetcher:
            Optional callable used by `add_urls`. Signature: (url: str) -> Optional[str].
        include_exts:
            Allowed file extensions for file/directory/zip ingestion. Defaults to
            `supported_file_types`.
        extra_metadata:
            Extra metadata merged into each queued document's metadata.
        reset_state:
            If True, clears previous queue and seen-hash set before processing.

        Returns
        -------
        Dict[str, Any]
            A summary dictionary, typically the result of `build()`. On error, returns:
                {
                    "status": "error",
                    "error": "<message>",
                    "unsupported": [...optional list of unsupported items...]
                }
        """
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
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        retrieval_mode: RetrievalMode = "classic",       
        retrieval_scope: Optional[RetrievalScope] = None,
        retrieval_plan: Optional[RetrievalPlan] = None,
        answer_breadth: Optional[AnswerBreadth] = None,
    ) -> RetrieveResult:
        """
        Public interface for retrieving documents from the vectorstore.
        
        Args:
            user_query: The user's query string.
            top_k: Optional number of documents to retrieve.
            metadata_filter: Optional metadata filter for the search.
            similarity_threshold: Optional similarity threshold for the search.
        
        Returns:
            RetrieveResult: The result of the retrieval process.
        """
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

        # --- Decide the plan ---
        if retrieval_plan is not None:
            plan = retrieval_plan
        elif retrieval_mode == "classic":
            plan = RetrievalPlan(
                mode="classic",
                scope=None,
                answer_breadth=None,
                top_k=top_k or self.cfg.top_k,
            )
        else:
            # smart mode
            if retrieval_scope is not None:
                # map scope -> default top_k if caller didn't provide one
                default_k = self._compute_top_k(retrieval_scope, answer_breadth, max_top_k=50)
                plan = RetrievalPlan(
                    mode="smart",
                    scope=retrieval_scope,
                    answer_breadth=answer_breadth,
                    top_k=default_k,
                )
            else:
                # let the agent plan using router_llm
                plan = self._plan_retrieval(user_query)
        
        self.logger.info(f"[RAGAgent.retrieve] Retrieval plan: {plan}")
        # Embed the query and retrieve documents based on the plan
        query_vec = self.embedder.embed(query=user_query, normalize_embeddings=self.cfg.normalize_embeddings)
        raw = self.vectorstore.similarity_search(
            query_embedding=query_vec,
            top_k=plan.top_k or self.cfg.top_k,
            metadata_filter=metadata_filter,
            similarity_threshold=similarity_threshold or self.cfg.similarity_threshold,
        )
        docs, distances = self._unpack_results(raw)
        self.logger.info(f"[RAGAgent.retrieve] Retrieved {len(docs)} documents")

        # Build + trim (only context, using a buffer for prompts/history)
        untrimmed_context = self.ctx.build(docs)
        self.logger.info(f"[RAGAgent.retrieve] Untrimmed context: {len(untrimmed_context)} chars")
        trim = self._trim_context_by_token_limit(
            db_context=untrimmed_context,
            reserved_prompt_tokens=getattr(self.cfg, "prompt_token_buffer", 500),
        )
        self.logger.info(f"[RAGAgent.retrieve] Trimmed context: {len(trim.trimmed_context)} chars")

        return RetrieveResult(
            context=trim.trimmed_context,
            max_new_tokens=trim.final_max_new_tokens,
            base_tokens=trim.base_tokens,  # now: reserved prompt/history tokens
            context_tokens_used=trim.context_tokens_used,
            token_budget=int(getattr(self.llm, "max_seq_length", 8192)),
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
        user_query: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        *,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **llm_kwargs: Any,
    ) -> LLM_RESPONSE_TYPE:
        """
        Public interface for running the RAG pipeline.
        
        Args:
            user_query: The user's query string.
            system_prompt: Optional system prompt for the LLM.
            chat_history: Optional chat history for the LLM.
            stream: Whether to stream the response.
            top_k: Optional number of documents to retrieve.
            metadata_filter: Optional metadata filter for the search.
            similarity_threshold: Optional similarity threshold for the search.
            temperature: Optional temperature for the LLM.
            **llm_kwargs: Additional keyword arguments for the LLM.
        Returns:
            LLM_RESPONSE_TYPE:
                - If stream=False: Returns ChatCompletionResponse (Pydantic model)
                - If stream=True: Returns Generator yielding ChatCompletionChunk objects
        Raises:
            ValueError: If vectorstore or embedder is not provided.
            ValueError: If LLM is not provided.
        """
        if not self.vectorstore or not self.embedder:
            raise ValueError("VectorDB and embedder must be provided to RAGAgent")
        if not self.llm:
            raise ValueError("LLM must be provided to RAGAgent. Please reinitialize the agent with a valid LLM instance.")
        
        retrieved: RetrieveResult = self.retrieve(
            user_query=user_query,
            chat_history=chat_history or [],
            top_k=top_k or self.cfg.top_k,
            metadata_filter=metadata_filter,
            similarity_threshold=similarity_threshold,
        )
        sys_prompt = system_prompt or RAG_AGENT_SYSTEM_PROMPT
        user_prompt = RAG_USER_PROMPT.format(context=retrieved.context, query=user_query)

        # Choose final generation cap
        return self.llm.ask(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            chat_history=chat_history or [],
            temperature=temperature or self.cfg.temperature,
            max_new_tokens=retrieved.max_new_tokens,
            stream=stream,
            **llm_kwargs,
        )

    def _plan_retrieval(
        self,
        user_query: str,
    ) -> RetrievalPlan:
        """
        Use router_llm to decide retrieval scope and top_k when in 'smart' mode.
        """
        if self.router_llm is None:
            # Fallback: simple heuristic
            return RetrievalPlan(
                mode="smart",
                scope="medium",
                answer_breadth="single_fact",
                top_k=self.cfg.top_k,
            )
        user_prompt = (
            f"User query:\n{user_query}\n"
            "Decide the retrieval scope and answer breadth for this question."
            "Remember: respond ONLY with a JSON object."
        )
        try:
            content = self.router_llm.ask(
                system_prompt=RETRIEVAL_PLANNER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=256,
                temperature=0.2,
                stream=False,
            ).choices[0].message.content

            obj = extract_and_repair_json(content, return_dict=True)
            scope = obj.get("scope", "medium")
            answer_breadth = obj.get("answer_breadth", "single_fact")

            top_k = self._compute_top_k(scope, answer_breadth)
            return RetrievalPlan(
                mode="smart",
                scope=scope,
                answer_breadth=answer_breadth,
                top_k=top_k,
            )
        except Exception:
            # Be robust: fall back to something sane
            self.logger.exception("RAGAgent._plan_retrieval failed; using default plan.")
            return RetrievalPlan(
                mode="smart",
                scope="medium",
                answer_breadth="single_fact",
                top_k=self.cfg.top_k,
            )

    def _compute_top_k(self, scope: str, breadth: str, max_top_k: int = 50) -> int:
        base_k = RETRIEVAL_SCOPE_BASE_K.get(scope, 10)
        mult = ANSWER_BREADTH_MULTIPLIER.get(breadth, 1)
        k = base_k * mult
        return min(k, max_top_k)
        
    # ---------- Internals ----------
    def _trim_context_by_token_limit(
        self,
        db_context: str,
        *,
        reserved_prompt_tokens: Optional[int] = None,
    ) -> _TrimResult:
        """
        Trim only the DB context using a fixed token buffer for prompts/history.

        We assume:
            reserved_prompt_tokens ≈ tokens(system + history + user)
        
        Ensure:
            reserved_prompt_tokens
            + tokens(trimmed_context)
            + max_new_tokens
            <= max_seq_length - safety_margin
        """
        max_seq = int(getattr(self.llm, "max_seq_length", 8192))
        max_ctx = max_seq - int(self.cfg.safety_margin_tokens)

        # How many tokens to reserve for system + user + history
        prompt_buffer = int(reserved_prompt_tokens or self.cfg.prompt_token_buffer)
        # Clamp buffer so we still have some room for output
        min_output_tokens = int(self.cfg.min_output_tokens)
        if prompt_buffer + min_output_tokens > max_ctx:
            # Just guarantee at least min_output_tokens for generation
            prompt_buffer = max(0, max_ctx - min_output_tokens)
        base_tokens = prompt_buffer  # tokens already "spent" on prompts/history
        # A) initial target for output tokens
        if self.cfg.fixed_max_new_tokens is not None:
            initial_max_new_tokens = int(self.cfg.fixed_max_new_tokens)
        else:
            remaining = max(max_ctx - base_tokens, 0)
            initial_max_new_tokens = max(
                int(remaining * self.cfg.auto_output_ratio),
                min_output_tokens,
            )
        # B) budget left for context
        available_for_context = max(max_ctx - base_tokens - initial_max_new_tokens, 0)
        # C) trim context to budget
        trimmed_context, context_tokens_used = self.tokens.trim_to_tokens(
            db_context, available_for_context
        )
        # D) recompute final output cap with actual context usage
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
            base_tokens=base_tokens,  # now: reserved prompt/history tokens
            context_tokens_used=context_tokens_used,
            available_for_context=available_for_context,
            initial_max_new_tokens=initial_max_new_tokens,
            final_max_new_tokens=final_max_new_tokens,
            generation_budget=generation_budget,
        )

    @staticmethod
    def _calculate_final_max_new_tokens(
        fixed_max_new_tokens: Optional[int],
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
        raw: Union[List[Doc], List[Tuple[Doc, float]]]
    ) -> Tuple[List[Doc], List[Optional[float]]]:
        docs: List[Doc] = []
        dists: List[Optional[float]] = []
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
