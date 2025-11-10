# neurosurfer/models/base.py
"""
Base Chat Model Module
======================

This module provides the abstract base class for all chat models in Neurosurfer.
It defines a unified interface for interacting with different LLM backends
(Transformers, Unsloth, vLLM, LlamaCpp, OpenAI) with consistent API.

The BaseModel class handles:
    - Unified chat completion interface (streaming and non-streaming)
    - OpenAI-compatible response formats using Pydantic models
    - Token counting and context window management
    - Chat template formatting
    - Stop word detection and thinking tag suppression
    - Thread-safe generation with stop signals

All concrete model implementations must inherit from BaseModel and implement:
    - init_model(): Initialize the underlying model
    - _call(): Non-streaming generation
    - _stream(): Streaming generation
    - stop_generation(): Interrupt ongoing generation
"""
import logging
import uuid
from threading import Lock
from typing import Any, Generator, List, Dict, Union, Optional, Tuple, Set, Type, Callable
from typing import get_origin, get_args
from datetime import datetime
import time
from abc import ABC, abstractmethod
import threading
from threading import RLock

from dataclasses import dataclass
import json
from pydantic import BaseModel as PydanticModel
from pydantic.json_schema import models_json_schema

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


class BaseModel(ABC):
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
        >>> class MyModel(BaseModel):
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
        self.model_name = "local-gpt"
        self.verbose = verbose
        self.logger = logger
        self.call_id = None
        self.lock = Lock()
        self.model = None
        self.max_seq_length = max_seq_length
        self.enable_thinking = enable_thinking
        self.stop_words = stop_words or []
        self._stop_signal = False
        self.lock = RLock()

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
        *,
        output_schema: Optional[Type[PydanticModel]] = None,
        strict_json: bool = True,
        on_parse_error: Optional[Callable[[str], str]] = None,
        max_repair_attempts: int = 1,
        **kwargs: Any,
    ) -> Union[
        ChatCompletionResponse,
        Generator[ChatCompletionChunk, None, None],
        PydanticModel,
    ]:
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

            output_schema (Type[PydanticModel]): Optional Pydantic model for returning structured responses.
            strict_json (bool): Whether to enforce strict JSON parsing. Default: True
            on_parse_error (Callable[[str], str]): Optional callback to handle parse errors.
            max_repair_attempts (int): Maximum number of repair attempts. Default: 1
        
        Returns:
            Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None], PydanticModel]:
                - If stream=False: Returns ChatCompletionResponse (Pydantic model)
                - If stream=True: Returns Generator yielding ChatCompletionChunk objects
                - If output_schema is provided: Returns PydanticModel. Structured Responses are always non-streaming.
        
        Example:
            >>> # Non-streaming
            >>> response = model.ask("What is AI?", temperature=0.5)
            >>> print(response.choices[0].message.content)
            
            >>> # Streaming
            >>> for chunk in model.ask("Explain quantum computing", stream=True):
            ...     print(chunk.choices[0].delta.content, end="")
            
            >>> # Structured
            >>> response = model.ask("Give me 3 examples of AI applications", output_schema=MyPydanticModel)
            >>> print(response.data)
        """

        self.call_id = str(uuid.uuid1())
        # Structured mode takes precedence; force non-streaming
        if output_schema is not None:
            if stream:
                self.logger.warning("[BaseModel] `output_schema` provided with `stream=True`; forcing non-streaming structured output.")
            # sys = self._make_structured_system_prompt(system_prompt, output_schema, strict_json=strict_json)
            sys = self._make_minimal_structured_system_prompt(system_prompt, output_schema)
            print(sys)
        
            resp: ChatCompletionResponse = self._call(
                user_prompt=user_prompt,
                system_prompt=sys,
                chat_history=chat_history or [],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            raw_text = resp.choices[0].message.content or ""
            
            candidate = self._extract_json_object(raw_text)
            print(candidate)

            if candidate is None and on_parse_error and max_repair_attempts > 0:
                try:
                    repaired = on_parse_error(raw_text)
                    candidate = self._extract_json_object(repaired)
                except Exception:
                    candidate = None

            if candidate is None:
                raise ValueError(
                    "Structured output: could not locate a JSON object in model output.\n"
                    f"--- RAW ---\n{raw_text}"
                )

            try:
                parsed = output_schema.model_validate_json(candidate)
            except Exception as e:
                if on_parse_error and max_repair_attempts > 0:
                    try:
                        repaired = on_parse_error(candidate)
                        parsed = output_schema.model_validate_json(repaired)
                    except Exception as e2:
                        raise ValueError(
                            f"Structured output JSON failed validation after repair: {e2}\n--- RAW ---\n{raw_text}"
                        ) from e2
                else:
                    raise ValueError(
                        f"Structured output JSON failed validation: {e}\n--- JSON ---\n{candidate}"
                    ) from e
            return parsed

        # Normal path (unchanged)
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

    def _delta_chunk(self, call_id: str, model: str, content: str) -> ChatCompletionChunk:
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

    def _stop_chunk(self, call_id: str, model: str, finish_reason: str = "stop") -> ChatCompletionChunk:
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

    def _final_nonstream_response(
        self, 
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

    def _format_messages(
        self,
        tokenizer,
        system_prompt: str,
        chat_history: List[dict],
        user_prompt: str,
        return_string: bool = False,
        return_list: bool = False,
    ) -> Union[str, List[Dict[str, str]]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})
        
        if return_string:
            return self._format_chat_to_string(messages)
        elif return_list:
            return messages
        else:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

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
                self.logger.info("[BaseModel] Stop signal detected.")
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
                self.logger.info(f"[BaseModel] Stop word '{hit}' detected.")
                break
            if piece: yield piece

    def _schema_as_pretty_json(self, schema_cls: Type[PydanticModel]) -> str:
        """
        Build JSON Schema for a single pydantic model (Pydantic v2).
        Works without tuple-packing or multi-model plumbing.
        """
        schema = schema_cls.model_json_schema(ref_template="#/components/schemas/{model}")  # v2 API
        # If you prefer the fully-inlined schema for the model itself:
        # many UIs do better with just the direct object schema for display
        return json.dumps(schema, ensure_ascii=False, indent=2)

    def _make_structured_system_prompt(
        self,
        base_system_prompt: str,
        schema_cls: Type[PydanticModel],
        *,
        strict_json: bool = True,
    ) -> str:
        schema_json = self._schema_as_pretty_json(schema_cls)
        rules = [
            "You MUST respond with a single JSON object that validates against the schema below.",
            "Do NOT include code fences, markdown, explanations, or additional keys.",
            "If a field is unknown, choose a default that still validates (e.g., empty string/array/0/false).",
        ]
        if strict_json:
            rules.append("Output MUST be valid JSON (RFC 8259). No comments or trailing commas.")
        sys = (base_system_prompt or "You are a function that returns JSON.").strip()
        return (
            f"{sys}\n\n"
            "## Structured Output Contract\n"
            + "\n".join(f"- {r}" for r in rules)
            + "\n\n### JSON Schema\n"
            f"{schema_json}\n"
        )

    def _extract_json_object(self, text: str) -> Optional[str]:
        if not text: return None
        t = text.strip()
        if t.startswith("{") and t.endswith("}"): return t
        # naive balanced-brace extractor (good enough; no extra deps)
        depth = 0
        start = None
        for i, ch in enumerate(t):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return t[start:i+1]
        return None


    # ---------- Minimal structural schema (prompt-side only) ----------

    def _py_type_to_minimal(self, t) -> str | dict | list:
        """
        Map Python/Pydantic field annotations to minimal structural markers.

        Returns one of:
          - "string" | "integer" | "number" | "boolean" | "object" | "array"
          - dict (nested object with properties)
          - list (single-item list meaning homogeneous array element type)
        """
        origin = get_origin(t)
        args = get_args(t)

        # Optional[X] or Union[X, None] -> use X
        if origin is Union and len(args) == 2 and type(None) in args:
            non_none = args[0] if args[1] is type(None) else args[1]
            return self._py_type_to_minimal(non_none)

        # Generic containers
        if origin in (list, List, tuple, Tuple, set, Set):
            elem = args[0] if args else str  # default to string
            return [self._py_type_to_minimal(elem)]

        if origin in (dict, Dict):
            # keep simple; object is enough for guidance
            return "object"

        # Primitive types
        if t in (str,):
            return "string"
        if t in (int,):
            return "integer"
        if t in (float,):
            return "number"
        if t in (bool,):
            return "boolean"

        # Pydantic model (nested object)
        try:
            # v2: Pydantic models have .model_fields
            fields = getattr(t, "model_fields", None)
            if isinstance(fields, dict):
                return self._minimal_object_from_model(t)
        except Exception:
            pass

        # Fallbacks
        return "object"

    def _minimal_object_from_model(self, schema_cls: Type[PydanticModel]) -> dict:
        """
        Build a compact schema for a Pydantic model:
        {
          "title": "Car",
          "type": "object",
          "required": [...],
          "properties": { k: <minimal type> | {nested} | [elem] }
        }
        """
        fields = schema_cls.model_fields  # pydantic v2
        required = [name for name, f in fields.items() if f.is_required()]
        props: dict[str, object] = {}

        for name, f in fields.items():
            ann = f.annotation
            props[name] = self._py_type_to_minimal(ann)

        return {
            "title": schema_cls.__name__,
            "type": "object",
            "required": required,
            "properties": props,
        }

    def _schema_as_minimal_structure(self, schema_cls: Type[PydanticModel]) -> str:
        """
        Produce a compact, human-readable schema that conveys only structure.
        """
        struct = self._minimal_object_from_model(schema_cls)
        # pretty but short; if you prefer single-line, set indent=None and separators=(',',':')
        return json.dumps(struct, ensure_ascii=False, indent=2)

    def _make_minimal_structured_system_prompt(
        self,
        base_system_prompt: str,
        schema_cls,
    ) -> str:
        """
        Embed the minimal schema plus concise rules.
        """
        sys = (base_system_prompt or "You are a function that returns JSON.").strip()
        minimal = self._schema_as_minimal_structure(schema_cls)
        rules = [
            "Return a single JSON object that matches the structure below.",
            "No code fences, no markdown, no explanations, no extra keys.",
            "The output MUST be valid JSON (RFC 8259).",
        ]
        return (
            f"{sys}\n\n"
            "## Structured Output Contract\n"
            + "\n".join(f"- {r}" for r in rules)
            + "\n\n### Structure\n"
            f"{minimal}\n"
        )
