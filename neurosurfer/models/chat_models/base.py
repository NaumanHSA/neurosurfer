# neurosurfer/models/base.py
"""
Base Chat Model Module
======================

This module provides the abstract base class for all chat models in Neurosurfer.
It defines a unified interface for interacting with different LLM backends
(Transformers, Unsloth, vLLM, LlamaCpp, OpenAI) with consistent API.

The BaseChatModel class handles:
    - Unified chat completion interface (streaming and non-streaming)
    - OpenAI-compatible response formats using Pydantic models
    - Token counting and context window management
    - Chat template formatting
    - Stop word detection and thinking tag suppression
    - Thread-safe generation with stop signals

All concrete model implementations must inherit from BaseChatModel and implement:
    - init_model(): Initialize the underlying model
    - _call(): Non-streaming generation
    - _stream(): Streaming generation
    - stop_generation(): Interrupt ongoing generation
"""
import logging
import uuid
from threading import Lock
from typing import Any, Generator, List, Dict, Union, Optional, Tuple, Set, Type, Callable
from datetime import datetime
import time
from abc import ABC, abstractmethod
from threading import RLock
from jinja2.exceptions import TemplateError

# Import Pydantic models
from neurosurfer.server.schemas import (
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChoiceMessage,
    StreamChoice,
    DeltaContent,
    Usage
)
from neurosurfer.config import config
from transformers import TextIteratorStreamer

LLM_RESPONSE_TYPE = Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]


class BaseChatModel(ABC):
    """
    Abstract base class for all chat models in Neurosurfer.
    
    This class provides a unified interface for interacting with different LLM backends
    while maintaining OpenAI-compatible response formats. All model implementations
    (Transformers, Unsloth, vLLM, LlamaCpp, OpenAI) inherit from this class.
    
    Attributes:
        model_name (str): Identifier for the model (default: "local-gpt")
        verbose (bool): Enable verbose logging
        logger (logging.Logger): Logger instance for debugging
        call_id (str): Unique identifier for each generation call
        lock (Lock): Thread lock for concurrent access control
        model: The underlying model instance (implementation-specific)
        max_seq_length (int): Maximum context window size in tokens
    
    Abstract Methods:
        init_model(): Initialize the underlying model and tokenizer
        _call(): Perform non-streaming generation
        _stream(): Perform streaming generation
        stop_generation(): Stop ongoing generation
    
    Example:
        >>> class MyModel(BaseChatModel):
        ...     def init_model(self):
        ...         # Load model
        ...         pass
        ...     
        ...     def _call(self, user_prompt, system_prompt, **kwargs):
        ...         # Generate response
        ...         return self._final_nonstream_response(...)
        ...     
        ...     def _stream(self, user_prompt, system_prompt, **kwargs):
        ...         # Stream response
        ...         for token in tokens:
        ...             yield self._delta_chunk(...)
        ...         yield self._stop_chunk(...)
    """
    def __init__(
        self,
        max_seq_length: int = config.base_model.max_seq_length,
        enable_thinking: bool = config.base_model.enable_thinking,
        stop_words: Optional[List[str]] = config.base_model.stop_words,
        verbose: bool = config.base_model.verbose,
        logger: logging.Logger = logging.getLogger(),
        **kwargs,
    ):
        """
        Initialize the base model.
        
        Args:
            max_seq_length (int): Maximum context window size in tokens. Default: 4096
            verbose (bool): Enable verbose logging. Default: False
            logger (logging.Logger): Logger instance for debugging
            **kwargs: Additional model-specific parameters
        """
        self.max_seq_length = max_seq_length
        self.stop_words = stop_words or []
        self.enable_thinking = enable_thinking
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)
        self.call_id = None
        self.lock = Lock()
        self._stop_signal = False

        # To be set by subclasses
        self.model_name: str = getattr(self, "model_name", "unknown-model")
        self.model = getattr(self, "model", None)
        self.tokenizer = getattr(self, "tokenizer", None)


    def set_stop_signal(self):
        """Set stop signal to halt generation"""
        with self.lock:
            self._stop_signal = True

    def reset_stop_signal(self):
        """Reset stop signal before new generation"""
        with self.lock:
            self._stop_signal = False

    @abstractmethod
    def init_model(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def ask(
        self,
        user_prompt: str,
        system_prompt: str = config.base_model.system_prompt,
        chat_history: List[dict] = [],
        temperature: float = config.base_model.temperature,
        max_new_tokens: int = config.base_model.max_new_tokens,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLM_RESPONSE_TYPE:
        """
        Main entry point for generating model responses.
        
        This method provides a unified interface for both streaming and non-streaming
        generation. It automatically routes to _call() or _stream() based on the
        stream parameter.
        
        Args:
            user_prompt (str): The user's input message/question
            system_prompt (str): System-level instructions for the model.
                Default: "You are a helpful assistant. Answer questions to the best of your ability."
            chat_history (List[dict]): Conversation history as list of message dicts.
                Each dict should have 'role' and 'content' keys. Default: []
            temperature (float): Sampling temperature (0.0-2.0). Lower = more deterministic,
                higher = more creative. Default: 0.7
            max_new_tokens (int): Maximum number of tokens to generate. Default: 2000
            stream (bool): Enable streaming response. Default: False
            **kwargs: Additional model-specific generation parameters
        
        Returns:
            LLM_RESPONSE_TYPE:
                - If stream=False: Returns ChatCompletionResponse (Pydantic model)
                - If stream=True: Returns Generator yielding ChatCompletionChunk objects
        
        Example:
            >>> # Non-streaming
            >>> response = model.ask("What is AI?", temperature=0.5)
            >>> print(response.choices[0].message.content)
            
            >>> # Streaming
            >>> for chunk in model.ask("Explain quantum computing", stream=True):
            ...     print(chunk.choices[0].delta.content, end="")            
        """

        self.call_id = str(uuid.uuid1())
        params = dict({
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "chat_history": chat_history,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            **kwargs
        })
        return self._stream(**params) if stream else self._call(**params)

    @abstractmethod
    def _call(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict] = [],
        temperature: float = config.base_model.temperature,
        max_new_tokens: int = config.base_model.max_new_tokens,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Must return ChatCompletionResponse Pydantic model"""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _stream(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict] = [],
        temperature: float = config.base_model.temperature,
        max_new_tokens: int = config.base_model.max_new_tokens,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Must yield ChatCompletionChunk Pydantic models"""
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def _delta_chunk(call_id: str, model: str, content: str) -> ChatCompletionChunk:
        """
        Create a streaming delta chunk (Pydantic model).
        
        This helper method constructs an OpenAI-compatible streaming chunk
        containing incremental content.
        
        Args:
            call_id (str): Unique identifier for this generation call
            model (str): Model identifier
            content (str): Incremental text content for this chunk
        
        Returns:
            ChatCompletionChunk: Pydantic model representing a streaming chunk
        """
        return ChatCompletionChunk(
            id=call_id,
            created=int(time.time()),
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(content=content),
                    finish_reason=None
                )
            ]
        )

    @staticmethod
    def _stop_chunk(call_id: str, model: str, finish_reason: str = "stop") -> ChatCompletionChunk:
        """
        Create a final streaming chunk indicating completion (Pydantic model).
        
        This helper method constructs the final chunk in a streaming response,
        signaling that generation is complete.
        
        Args:
            call_id (str): Unique identifier for this generation call
            model (str): Model identifier
            finish_reason (str): Reason for completion. Options: "stop", "length", "error".
                Default: "stop"
        
        Returns:
            ChatCompletionChunk: Final streaming chunk with empty delta and finish_reason
        """
        return ChatCompletionChunk(
            id=call_id,
            created=int(time.time()),
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(),  # Empty delta for stop chunk
                    finish_reason=finish_reason
                )
            ]
        )

    @staticmethod
    def _final_nonstream_response(
        call_id: str, 
        model: str, 
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> ChatCompletionResponse:
        """
        Create a complete non-streaming response (Pydantic model).
        
        This helper method constructs an OpenAI-compatible chat completion response
        with the full generated content and token usage statistics.
        
        Args:
            call_id (str): Unique identifier for this generation call
            model (str): Model identifier
            content (str): Complete generated text
            prompt_tokens (int): Number of tokens in the prompt. Default: 0
            completion_tokens (int): Number of tokens in the completion. Default: 0
        
        Returns:
            ChatCompletionResponse: Complete response with content and usage stats
        """
        return ChatCompletionResponse(
            id=call_id,
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=content),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    # ------------------------------------------------------------------
    # Message construction & formatting
    # ------------------------------------------------------------------
    def _normalize_for_chat_template(
        self,
        *,
        system_prompt: Optional[str],
        chat_history: List[Dict[str, Any]],
        user_prompt: str,
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """
        Normalize conversation for chat_template.

        - Ensures final user message is part of the message list.
        - You can extend this to:
            * merge multiple system messages
            * drop tool messages
            * enforce allowed roles, etc.

        Returns:
            (normalized_system_prompt, normalized_history_messages)
            where `normalized_history_messages` ALREADY includes the final
            user message.
        """
        system_prompt = system_prompt or ""
        history: List[Dict[str, str]] = []

        # Make a shallow copy of existing history
        if chat_history:
            for m in chat_history:
                # assume well-formed {role, content}
                role = m.get("role", "user")
                content = m.get("content", "")
                history.append({"role": role, "content": content})

        # Append the current user prompt as the last user message
        history.append({"role": "user", "content": user_prompt})

        return system_prompt, history

    def _build_messages(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, Any]],
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        """
        Build a list of {'role', 'content'} messages in OpenAI style.

        This is the single source of truth for how messages are constructed.
        All backends (Unsloth, Transformers, vLLM, etc.) should build on this.
        """

        # Qwen thinking toggle (works for qwen/qwen3/etc.)
        if "qwen" in self.model_name.lower() and not self.enable_thinking:
            system_prompt = (system_prompt or "") + "/nothink"

        system_prompt, normalized_history = self._normalize_for_chat_template(
            system_prompt=system_prompt,
            chat_history=chat_history or [],
            user_prompt=user_prompt,
        )

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(normalized_history)
        return messages

    def _supports_tokenizer_param(self, param: str) -> bool:
        """
        Check if tokenizer's chat_template references a given jinja variable,
        e.g. 'reasoning_effort'. If it's not in the template, passing it as a
        kwarg will raise a Jinja UndefinedError, so we skip it.
        """
        if not getattr(self, "tokenizer", None):
            return False
        template = getattr(self.tokenizer, "chat_template", "") or ""
        return param in template

    def _fallback_plain_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Simple, readable fallback formatting if chat_template fails or doesn't exist.

        Produces prompts like:

            [SYSTEM]
            ...
            [USER]
            ...
            [ASSISTANT]

        So generation naturally continues as the assistant.
        """
        system_prompt: Optional[str] = None
        rest: List[Dict[str, str]] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system" and system_prompt is None:
                system_prompt = content
            else:
                rest.append({"role": role, "content": content})

        parts: List[str] = []
        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}\n")

        for m in rest:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"[{role}]\n{content}\n")

        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)

    def _apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool = True,
        **template_kwargs: Any,   # <─ NEW: params from subclass
    ):
        """
        Centralized wrapper around tokenizer.apply_chat_template.

        - If tokenizer has a chat_template, use it.
        - Accepts arbitrary kwargs from subclasses (e.g. reasoning_effort="low").
        - For each kw:
            * If it's a Python-level param we know (e.g. enable_thinking), we
            handle it explicitly.
            * If it's a Jinja variable and the template references it, we pass it.
            * Otherwise we drop it (to avoid weird side effects).
        - On TypeError (e.g. tokenizer does not accept enable_thinking), we retry
        without that param.
        - On TemplateError or no chat_template → fall back to plain text.

        Returns either:
            - tokenized tensors (if tokenize=True)
            - or a raw prompt string (if tokenize=False)
        """
        if not getattr(self, "tokenizer", None):
            # No tokenizer? Only plain text makes sense.
            text = self._fallback_plain_prompt(messages)
            if tokenize:
                raise RuntimeError("Tokenizer is not set; cannot tokenize inputs.")
            return text

        tok = self.tokenizer
        has_chat_template = hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None)

        # Base kwargs that always go to apply_chat_template
        base_kwargs: Dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
        }
        if tokenize:
            base_kwargs.update({
                "return_tensors": "pt",
                "return_dict": True,
            })
        else:
            base_kwargs["tokenize"] = False

        # -----------------------------
        # Split incoming template_kwargs
        # -----------------------------
        # 1) Python-side known params (currently only enable_thinking)
        enable_thinking_val = template_kwargs.pop("enable_thinking", self.enable_thinking)

        # 2) Jinja context variables (reasoning_effort, target_length, etc.)
        #    Only pass them if the template actually uses them.
        jinja_kwargs: Dict[str, Any] = {}
        for key, value in template_kwargs.items():
            if self._supports_tokenizer_param(key):
                jinja_kwargs[key] = value
            else:
                # Not referenced in template, safely ignore
                if self.verbose:
                    self.logger.debug(
                        "Dropping chat_template kwarg '%s'; not referenced in template.",
                        key,
                    )

        call_kwargs = {**base_kwargs, **jinja_kwargs}

        # -----------------------------
        # Main path: use chat_template
        # -----------------------------
        if has_chat_template:
            try:
                # First try with enable_thinking if we have a value
                return tok.apply_chat_template(
                    messages,
                    enable_thinking=enable_thinking_val,
                    **call_kwargs,
                )
            except TypeError as e:
                # Tokenizer does not accept `enable_thinking` as a Python arg.
                # Retry WITHOUT enable_thinking but keep the Jinja kwargs.
                self.logger.warning(
                    "apply_chat_template does not accept 'enable_thinking'; "
                    "retrying without it. Error: %s",
                    e,
                )
                try:
                    return tok.apply_chat_template(
                        messages,
                        **call_kwargs,
                    )
                except TemplateError as e2:
                    self.logger.warning(
                        "Chat template failed (%s). Falling back to plain formatting.",
                        e2,
                    )
            except TemplateError as e:
                self.logger.warning(
                    "Chat template failed (%s). Falling back to plain formatting.",
                    e,
                )
            # Fall through to fallback if we hit any of the above

        # -----------------------------
        # Fallback path: plain prompt
        # -----------------------------
        text = self._fallback_plain_prompt(messages)
        if tokenize:
            return tok(text, return_tensors="pt", return_dict=True)
        return text


    def _format_messages(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, Any]],
        user_prompt: str,
        enable_thinking: bool = False,
    ) -> str:
        """
        DEFAULT: build messages and return a single prompt string.

        Backends like `TransformersModel` can call this and then feed the prompt
        into `tokenizer(prompt, return_tensors="pt")` if they don't want to use
        chat_template tokenization directly.
        """
        messages = self._build_messages(system_prompt, chat_history, user_prompt)
        prompt_text = self._apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        assert isinstance(prompt_text, str), "Expected prompt_text to be a string."
        return prompt_text

    def _format_chat_to_string(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return prompt
    
    def stop_generation(self):
        return NotImplementedError("Subclasses must implement this method.")

    def _find_first_stop(self, text: str, stops: List[str]) -> Tuple[Optional[str], Optional[int]]:
        """
        Find the first occurrence of any stop word in text.
        
        Returns:
            Tuple of (matched_stop_word, index_where_it_starts) or (None, None)
        
        Optimized to short-circuit on first match.
        """
        if not stops:
            return None, None
        earliest = None
        which = None
        for s in stops:
            if not s:
                continue
            i = text.find(s)
            if i != -1 and (earliest is None or i < earliest):
                earliest = i
                which = s
        return (which, earliest) if which is not None else (None, None)
    
    # ---------- Stream Consumer with Stop Word & Thinking Support ----------
    def _transformers_consume_stream(self, streamer: TextIteratorStreamer) -> Generator:
        """
        Consume tokens from streamer with:
        - Stop signal enforcement (immediate halt)
        - Stop word detection (truncate before match)
        - Thinking tag suppression (when disabled)
        
        Args:
            streamer: TextIteratorStreamer instance
            
        Yields:
            str: Incremental text chunks
            
        Returns:
            str: Full aggregated text
        """
        rolling = ""      # Buffer for stop-word scanning
        in_think = False  # Track if we're inside <think> tags
        first_word = False
        for token in streamer:
            # Check stop signal
            if self._stop_signal:
                self.logger.info("[BaseChatModel] Stop signal detected.")
                break
            piece: str = token
            # Handle thinking tag suppression
            if not self.enable_thinking:
                # Track <think> tags and strip content between them
                if "<think>" in piece:
                    in_think = True
                    piece = piece.replace("<think>", "")
                if "</think>" in piece:
                    in_think = False
                    piece = piece.replace("</think>", "")
                # Skip content inside thinking tags
                if in_think: continue
                # Clean any remaining fragments
                if not rolling and not piece.strip(): continue
            rolling += piece
            # Check for stop words in rolling buffer
            hit, cutoff = self._find_first_stop(rolling, self.stop_words)
            if hit is not None:
                # Truncate before the stop sequence
                to_emit = rolling[:cutoff]
                if to_emit: yield to_emit
                self.logger.info(f"[BaseChatModel] Stop word '{hit}' detected.")
                break
            if piece: yield piece

    def silent(self):
        if self.logger:
            self.logger.disabled = True


    def _normalize_for_chat_template(
        self,
        system_prompt: Optional[str],
        chat_history: List[dict],
        user_prompt: str,
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """
        Make the conversation safe for strict chat templates:

        - Merge any system messages from history into the main system_prompt.
        - Drop or map unsupported roles (tool/function/etc.).
        - Enforce user/assistant alternation as best as possible.
        - Avoid double 'user' at the end when we append user_prompt.
        """

        # 1) Merge extra system messages from history into system_prompt
        extra_system_parts = [
            (m.get("content") or "")
            for m in chat_history
            if m.get("role") == "system" and m.get("content")
        ]
        if extra_system_parts:
            merged = "\n\n".join(extra_system_parts)
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{merged}"
            else:
                system_prompt = merged

        # 2) Filter / map roles to only 'user' and 'assistant'
        filtered: List[Dict[str, str]] = []
        for m in chat_history:
            role = m.get("role")
            content = m.get("content") or ""
            if not content:
                continue

            if role in ("user", "assistant"):
                filtered.append({"role": role, "content": content})
            elif role in ("tool", "function"):
                # Treat tool outputs as assistant text (common pattern)
                filtered.append({"role": "assistant", "content": content})
            # else: drop unknown roles

        # 3) Enforce alternation user/assistant/user/assistant/...
        normalized: List[Dict[str, str]] = []
        expected_role = "user"

        for m in filtered:
            role = m["role"]
            content = m["content"]

            if not normalized:
                # First non-system message must be user; if not, coerce.
                if role != "user":
                    role = "user"
                normalized.append({"role": role, "content": content})
                expected_role = "assistant"
                continue

            # For subsequent messages, either match expected role or merge
            if role == expected_role:
                normalized.append({"role": role, "content": content})
                expected_role = "user" if expected_role == "assistant" else "assistant"
            else:
                # Same role twice in a row → merge into previous message
                normalized[-1]["content"] += "\n\n" + content
                # expected_role stays as-is

        # 4) Append current user prompt safely
        user_prompt = user_prompt or ""
        if user_prompt:
            if normalized and normalized[-1]["role"] == "user":
                # Avoid user/user; merge
                normalized[-1]["content"] += "\n\n" + user_prompt
            else:
                normalized.append({"role": "user", "content": user_prompt})

        return system_prompt, normalized
        