from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import logging
from collections import defaultdict

from altair import RadialGradient


from ..vectorstores.base import Doc, BaseVectorDB
from ..models.chat_models.base import BaseModel
from ..models.embedders import BaseEmbedder

# ---------- Results container ----------

@dataclass
class RetrieveResult:
    base_system_prompt: str
    base_user_prompt: str                      # user prompt with {context} filled in
    context: str                          # trimmed context actually used
    max_new_tokens: int                   # final dynamic value
    base_tokens: int                      # tokens for system+history+user (no context)
    context_tokens_used: int              # tokens consumed by context after trimming
    token_budget: int                     # total model window
    generation_budget: int                # remaining tokens available for generation
    docs: List[Doc] = field(default_factory=list)
    distances: List[Optional[float]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)  # extra debug/trace info


# ---------- Agent ----------
class RAGRetrieverAgent:
    """
    RAGRetrieverAgent:
      - Vector DB-agnostic: relies on your BaseVectorDB interface.
      - Embedder-agnostic: any object exposing .embed([str], normalize_embeddings=bool) -> [[float]].
      - LLM/tokenizer-agnostic: needs llm.tokenizer and llm.max_seq_length.

    Core features:
      1) Retrieves top-K docs (handles (Doc, distance) tuples).
      2) Builds a joined text context with an overridable formatter.
      3) Trims context to fit model window.
      4) Dynamically computes max_new_tokens after trimming.

    Usage:
      result = agent.retrieve(
          user_query="...",
          prompt={"system_prompt": "...", "user_prompt": "Answer:\n{context}\n\nQuestion: {query}"},
          chat_history=[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}],
          top_k=6,
      )
      -> result.user_prompt, result.system_prompt, result.context, result.max_new_tokens, ...
    """

    def __init__(
        self,
        llm: BaseModel,
        vectorstore: BaseVectorDB,
        embedder: BaseEmbedder,
        *,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        # Output budgeting
        fixed_max_new_tokens: Optional[int] = None,     # if set, used as initial cap; final is <= remaining budget
        auto_output_ratio: float = 0.25,                # when fixed_max_new_tokens is None, allocate this ratio initially
        min_output_tokens: int = 32,                    # ensure at least this many tokens for generation
        safety_margin_tokens: int = 32,                 # extra safety to avoid off-by-some in tokenization
        # Context formatting
        include_metadata_in_context: bool = True,
        context_separator: str = "\n\n---\n\n",
        context_item_header_fmt: str = "Source: {source}",
        make_source: Optional[Callable[[Doc], str]] = None,  # how to render a single doc's "source" line
        normalize_embeddings: bool = True,
    ):
        self.vector_db = vectorstore
        self.embedder = embedder
        self.llm = llm
        self.default_top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.fixed_max_new_tokens = fixed_max_new_tokens
        self.auto_output_ratio = auto_output_ratio
        self.min_output_tokens = min_output_tokens
        self.safety_margin_tokens = safety_margin_tokens
        self.include_metadata_in_context = include_metadata_in_context
        self.context_separator = context_separator
        self.context_item_header_fmt = context_item_header_fmt
        self.make_source = make_source or self._default_source
        self.normalize_embeddings = normalize_embeddings
        self.logger = logger or logging.getLogger(__name__)

    # ---------- Public API ----------
    def retrieve(
        self,
        user_query: str,
        base_system_prompt: str = "",
        base_user_prompt: str = "",
        chat_history: List[Dict[str, str]] = None,
        *,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RetrieveResult:
        """
        High-level single call:
          1) embed -> similarity_search
          2) build context string
          3) trim to fit + compute dynamic max_new_tokens
        """
        # 1) Embed query
        query_vec = self.embedder.embed(query=[user_query], normalize_embeddings=self.normalize_embeddings)[0]
        # 2) Retrieve
        raw = self.vector_db.similarity_search(
            query_embedding=query_vec,
            top_k=top_k or self.default_top_k,
            metadata_filter=metadata_filter,
            similarity_threshold=similarity_threshold or self.similarity_threshold,
        )

        docs, distances = self._unpack_results(raw)
        self.logger.info(f"[RAGRetriever] Retrieved {len(docs)} chunks")

        # 3) Build context (untrimmed)
        untrimmed_context = self._create_context(docs)
        trim = RAGRetrieverAgent._trim_context_by_token_limit(
            llm=self.llm,
            system_prompt=base_system_prompt,
            user_prompt=base_user_prompt,
            chat_history=chat_history,
            db_context=untrimmed_context,
            safety_margin_tokens=self.safety_margin_tokens,
            fixed_max_new_tokens=self.fixed_max_new_tokens,
            auto_output_ratio=self.auto_output_ratio,
            min_output_tokens=self.min_output_tokens,
        )

        result = RetrieveResult(
            base_system_prompt=base_system_prompt,
            base_user_prompt=base_user_prompt,
            context=trim.trimmed_context,
            max_new_tokens=trim.final_max_new_tokens,
            base_tokens=trim.base_tokens,
            context_tokens_used=trim.context_tokens_used,
            token_budget=self.llm.max_seq_length,
            generation_budget=trim.generation_budget,
            docs=docs,
            distances=distances,
            meta={
                "available_for_context": trim.available_for_context,
                "initial_max_new_tokens": trim.initial_max_new_tokens,
                "safety_margin_tokens": self.safety_margin_tokens,
            },
        )
        return result

    def _create_context(self, docs: List[Doc]) -> str:
        """
        Join retrieved docs into a single context string. Override for custom formatting.
        """
        parts: List[str] = []
        for d in docs:
            piece = d.text or ""
            if self.include_metadata_in_context:
                source = self.make_source(d)
                if source:
                    piece = f"{self.context_item_header_fmt.format(source=source)}\n{piece}"
            parts.append(piece.strip())
        return self.context_separator.join([p for p in parts if p])


    # ---------- Internals ----------
    @dataclass
    class _TrimResult:
        trimmed_context: str
        base_tokens: int
        context_tokens_used: int
        available_for_context: int
        initial_max_new_tokens: int
        final_max_new_tokens: int
        generation_budget: int
    
    @staticmethod
    def _trim_context_by_token_limit(
        llm: BaseModel,
        system_prompt: str,
        user_prompt: str,            # WITHOUT context inserted yet
        db_context: str,
        safety_margin_tokens: int = 32,
        chat_history: List[Dict[str, str]] = [],
        fixed_max_new_tokens: Optional[int] = None,
        auto_output_ratio: float = 0.25,                # when fixed_max_new_tokens is None, allocate this ratio initially
        min_output_tokens: int = 32,                    # ensure at least this many tokens for generation   
    ) -> _TrimResult:
        """
        Ensures: tokens(system + history + user + trimmed_context) + max_new_tokens <= max_seq_length - safety_margin
                 If fixed_max_new_tokens is None, uses auto ratio initially and then tightens to remaining budget.
        """
        max_ctx = int(llm.max_seq_length) - int(safety_margin_tokens)

        # Build base chat without context
        base_messages = []
        if system_prompt:
            base_messages.append({"role": "system", "content": system_prompt})
        base_messages.extend(chat_history or [])
        base_messages.append({"role": "user", "content": user_prompt.rstrip() + "\n\n"})

        base_prompt = RAGRetrieverAgent._apply_chat_template(llm, base_messages)
        base_tokens = RAGRetrieverAgent._count_tokens(llm, base_prompt)

        # Step A: choose an initial max_new_tokens
        if fixed_max_new_tokens is not None:
            initial_max_new_tokens = int(fixed_max_new_tokens)
        else:
            # allocate a portion of the remaining space to output as an initial target
            remaining = max(max_ctx - base_tokens, 0)
            initial_max_new_tokens = max(int(remaining * auto_output_ratio), min_output_tokens)

        # Step B: compute budget available for context (given the initial output target)
        available_for_context = max(max_ctx - base_tokens - initial_max_new_tokens, 0)

        # Step C: trim the context to this budget
        trimmed_context, context_tokens_used = RAGRetrieverAgent._trim_text_to_tokens(llm, db_context, available_for_context)

        # Step D: recompute the FINAL max_new_tokens using *actual* used context
        final_max_new_tokens = RAGRetrieverAgent._calculate_final_max_new_tokens(
            fixed_max_new_tokens=fixed_max_new_tokens,
            min_output_tokens=min_output_tokens,
            base_tokens=base_tokens,
            context_tokens_used=context_tokens_used,
            max_ctx=max_ctx,
        )

        # Generation budget = whatever remains after base+context
        generation_budget = max(max_ctx - base_tokens - context_tokens_used, 0)
        return RAGRetrieverAgent._TrimResult(
            trimmed_context=trimmed_context,
            base_tokens=base_tokens,
            context_tokens_used=context_tokens_used,
            available_for_context=available_for_context,
            initial_max_new_tokens=initial_max_new_tokens,
            final_max_new_tokens=final_max_new_tokens,
            generation_budget=generation_budget,
        )

    @staticmethod
    def _calculate_max_new_tokens_prompts(
        llm: BaseModel,
        system_prompt: str,
        user_prompt: str,
        chat_history: List[Dict[str, str]] = [],
        safety_margin_tokens: int = 32,
        fixed_max_new_tokens: Optional[int] = None,
        min_output_tokens: int = 32,
    ) -> int:
        max_ctx = int(llm.max_seq_length) - int(safety_margin_tokens)
        # Build base chat without context
        base_messages = []
        if system_prompt:
            base_messages.append({"role": "system", "content": system_prompt})
        base_messages.extend(chat_history or [])
        base_messages.append({"role": "user", "content": user_prompt.rstrip() + "\n\n"})
        base_prompt = RAGRetrieverAgent._apply_chat_template(llm, base_messages)
        base_tokens = RAGRetrieverAgent._count_tokens(llm, base_prompt)
        return RAGRetrieverAgent._calculate_final_max_new_tokens(
            fixed_max_new_tokens=fixed_max_new_tokens,
            min_output_tokens=min_output_tokens,
            base_tokens=base_tokens,
            context_tokens_used=0,
            max_ctx=max_ctx,
        )
   
    @staticmethod
    def _calculate_final_max_new_tokens(
        fixed_max_new_tokens: Optional[int], 
        min_output_tokens: int, 
        base_tokens: int, 
        context_tokens_used: int, 
        max_ctx: int
    ) -> int:
        """
        After context is trimmed, we can tighten max_new_tokens to the true remaining budget.
        If fixed_max_new_tokens was provided, we still must not exceed what's left.
        """
        remaining = max(max_ctx - base_tokens - context_tokens_used, 0)
        if fixed_max_new_tokens is not None:
            return max(min(fixed_max_new_tokens, remaining), 0)
        # If auto, we simply use the full remaining (but never below min_output_tokens)
        return max(remaining, min_output_tokens)

   
    # Default "source" rendering based on metadata
    @staticmethod
    def _default_source(d: Doc) -> str:
        md = d.metadata or {}
        return (
            md.get("filename")
            or md.get("source")
            or md.get("doc_id")
            or d.id
            or ""
        )

    # Token accounting helpers
    @staticmethod
    def _apply_chat_template(llm: BaseModel, messages: List[Dict[str, str]]) -> str:
        """
        Produce a single string via the model's chat template.
        Assumes tokenizer.apply_chat_template exists; otherwise fall back to naive concat.
        """
        tok = llm.tokenizer
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
        except Exception:
            # Fallback: simple concat; token counts may differ slightly
            buf = []
            for m in messages:
                role = m.get("role", "user").upper()
                buf.append(f"{role}:\n{m.get('content','')}\n")
            buf.append("ASSISTANT:\n")
            return "\n".join(buf)

    @staticmethod
    def _count_tokens(llm: BaseModel, text: str) -> int:
        tok = llm.tokenizer
        try:
            # Fast path used in your snippet
            return int(tok(text, return_tensors="pt").input_ids.shape[1])  # type: ignore[attr-defined]
        except Exception:
            try:
                # Generic path
                return len(tok.encode(text))  # type: ignore[attr-defined]
            except Exception:
                # Last resort
                return len(text.split())

    @staticmethod
    def _trim_text_to_tokens(llm: BaseModel, text: str, max_tokens: int) -> Tuple[str, int]:
        if max_tokens <= 0 or not text:
            return "", 0
        tok = llm.tokenizer
        try:
            ids = tok(text, return_tensors="pt").input_ids[0]  # type: ignore[attr-defined]
            used = min(len(ids), max_tokens)
            trimmed = tok.decode(ids[:used], skip_special_tokens=True)  # type: ignore[attr-defined]
            return trimmed.strip(), used
        except Exception:
            # Fallback tokenization by splitting to words—approximate
            words = text.split()
            # greedy shrink until token estimate fits
            lo, hi, best = 0, len(words), ""
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = " ".join(words[:mid])
                if RAGRetrieverAgent._count_tokens(llm, candidate) <= max_tokens:
                    best = candidate
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best.strip(), RAGRetrieverAgent._count_tokens(llm, best)

    @staticmethod
    def _unpack_results(
        raw: Union[List[Doc], List[Tuple[Doc, float]]]
    ) -> Tuple[List[Doc], List[Optional[float]]]:
        docs: List[Doc] = []
        distances: List[Optional[float]] = []
        if not raw:
            return docs, distances
        first = raw[0]
        if isinstance(first, tuple):
            for d, dist in raw:  # type: ignore[misc]
                docs.append(d)
                distances.append(dist)
        else:
            docs = raw  # type: ignore[assignment]
            distances = [None] * len(docs)
        return docs, distances


    def pick_files_by_grouped_chunk_hits(
        self,
        section_query: str,
        candidate_pool_size: int = 200,
        n_files: int = 10,
        file_key: str = "filename",
    ) -> List[str]:
        """
        Pick files most relevant to a specific query if chunks are from large number of files e.g. codebase.        
        Args:
            section_query (str): The query to search for.
            candidate_pool_size (int, optional): The number of candidate files to consider. Defaults to 200.
            n_files (int, optional): The number of files to return. Defaults to 10.
            file_key (str, optional): The key to use for file names in the metadata. Defaults to "filename".
        Returns:
            List[str]: A list of file paths most relevant to the query.
        """
        qemb = self.embedder.embed(section_query)  # shape: (d,)
        # broad search (no filter) to get signal across the codebase
        hits: List[Tuple[Doc, float]] = self.vector_db.similarity_search(query_embedding=qemb, top_k=candidate_pool_size)
        by_file = defaultdict(float)
        for doc, sim in hits:
            fp = doc.metadata.get(file_key)
            if not fp: 
                continue
            # sum sims; you can also take max or a weighted combo
            by_file[fp] += sim
        # sort by score desc and take top N
        return [fp for fp, _ in sorted(by_file.items(), key=lambda kv: kv[1], reverse=True)[:n_files]]
