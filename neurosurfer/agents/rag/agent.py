from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from neurosurfer.vectorstores.base import Doc, BaseVectorDB
from neurosurfer.models.chat_models.base import BaseModel
from neurosurfer.models.embedders.base import BaseEmbedder

from .config import RAGRetrieveConfig, RetrieveResult
from .token_utils import TokenCounter
from .context_builder import ContextBuilder


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
        llm: BaseModel,
        vectorstore: BaseVectorDB = None,
        embedder: BaseEmbedder = None,
        *,
        config: Optional[RAGRetrieveConfig] = None,
        logger: Optional[logging.Logger] = None,
        make_source=None,
    ):
        self.llm = llm
        self.vector_db = vectorstore
        self.embedder = embedder
        self.cfg = config or RAGRetrieveConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.tokens = TokenCounter(llm, chars_per_token=self.cfg.approx_chars_per_token)
        self.ctx = ContextBuilder(
            include_metadata_in_context=self.cfg.include_metadata_in_context,
            context_separator=self.cfg.context_separator,
            context_item_header_fmt=self.cfg.context_item_header_fmt,
            make_source=make_source,
        )

    # ---------- Public API ----------

    def retrieve(
        self,
        user_query: str,
        base_system_prompt: str = "",
        base_user_prompt: str = "",
        chat_history: Optional[List[Dict[str, str]]] = None,
        *,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RetrieveResult:
        """
        1) embed -> similarity_search
        2) build context
        3) trim to fit -> compute final max_new_tokens
        """
        if not self.vector_db or not self.embedder:
            raise ValueError("VectorDB and embedder must be provided to RAGAgent")
        # 1) Embed
        query_vec = self.embedder.embed(
            query=[user_query], normalize_embeddings=self.cfg.normalize_embeddings
        )[0]
        # 2) Retrieve
        raw = self.vector_db.similarity_search(
            query_embedding=query_vec,
            top_k=top_k or self.cfg.top_k,
            metadata_filter=metadata_filter,
            similarity_threshold=similarity_threshold or self.cfg.similarity_threshold,
        )
        docs, distances = self._unpack_results(raw)
        self.logger.info(f"[RAGRetriever] Retrieved {len(docs)} chunks")

        # 3) Build + trim
        untrimmed_context = self.ctx.build(docs)
        trim = self._trim_context_by_token_limit(
            system_prompt=base_system_prompt,
            user_prompt=base_user_prompt,
            chat_history=chat_history or [],
            db_context=untrimmed_context,
        )

        return RetrieveResult(
            base_system_prompt=base_system_prompt,
            base_user_prompt=base_user_prompt,
            context=trim.trimmed_context,
            max_new_tokens=trim.final_max_new_tokens,
            base_tokens=trim.base_tokens,
            context_tokens_used=trim.context_tokens_used,
            token_budget=int(getattr(self.llm, "max_seq_length", 8192)),
            generation_budget=trim.generation_budget,
            docs=docs,
            distances=distances,
            meta={
                "available_for_context": trim.available_for_context,
                "initial_max_new_tokens": trim.initial_max_new_tokens,
                "safety_margin_tokens": self.cfg.safety_margin_tokens,
            },
        )

    def run(
        self,
        user_query: str,
        base_system_prompt: str = "",
        base_user_prompt: str = "You are helpful.\n\nContext:\n{context}\n\nQuestion: {query}",
        chat_history: Optional[List[Dict[str, str]]] = None,
        *,
        stream: bool = True,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **llm_kwargs: Any,
    ):
        """
        Convenience method: perform retrieve and call the LLM with the filled prompt.
        When stream=True, yields chunks from llm.ask(…, stream=True). Otherwise returns the full response text.
        """
        rr = self.retrieve(
            user_query=user_query,
            base_system_prompt=base_system_prompt,
            base_user_prompt=base_user_prompt,
            chat_history=chat_history or [],
            top_k=top_k,
            metadata_filter=metadata_filter,
            similarity_threshold=similarity_threshold,
        )

        filled_user_prompt = rr.base_user_prompt.format(
            context=rr.context, query=user_query
        )

        # Choose final generation cap
        final_max_new = max_new_tokens if (max_new_tokens is not None) else rr.max_new_tokens
        temp = temperature if (temperature is not None) else 0.3

        if stream:
            gen = self.llm.ask(
                user_prompt=filled_user_prompt,
                system_prompt=rr.base_system_prompt,
                chat_history=chat_history or [],
                temperature=temp,
                max_new_tokens=final_max_new,
                stream=True,
                **llm_kwargs,
            )
            for chunk in gen:
                # forward through (assumes OpenAI-like delta.content)
                yield chunk.choices[0].delta.content or ""
            return
        else:
            resp = self.llm.ask(
                user_prompt=filled_user_prompt,
                system_prompt=rr.base_system_prompt,
                chat_history=chat_history or [],
                temperature=temp,
                max_new_tokens=final_max_new,
                stream=False,
                **llm_kwargs,
            )
            return resp.choices[0].message.content

    # ---------- Internals ----------
    def _trim_context_by_token_limit(
        self,
        system_prompt: str,
        user_prompt: str,
        db_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> _TrimResult:
        """
        Ensure: tokens(system + history + user + trimmed_context) + max_new_tokens
                <= max_seq_length - safety_margin
        """
        max_seq = int(getattr(self.llm, "max_seq_length", 8192))
        max_ctx = max_seq - int(self.cfg.safety_margin_tokens)

        # Build base chat without context
        base_messages = []
        if system_prompt:
            base_messages.append({"role": "system", "content": system_prompt})
        base_messages.extend(chat_history or [])
        base_messages.append({"role": "user", "content": user_prompt.rstrip() + "\n\n"})

        base_prompt = self.tokens.apply_chat_template(base_messages)
        base_tokens = self.tokens.count(base_prompt)

        # A) initial target for output tokens
        if self.cfg.fixed_max_new_tokens is not None:
            initial_max_new_tokens = int(self.cfg.fixed_max_new_tokens)
        else:
            remaining = max(max_ctx - base_tokens, 0)
            initial_max_new_tokens = max(
                int(remaining * self.cfg.auto_output_ratio),
                int(self.cfg.min_output_tokens),
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
            min_output_tokens=self.cfg.min_output_tokens,
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
