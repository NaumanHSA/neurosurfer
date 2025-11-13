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
from datetime import datetime
import time
from abc import ABC, abstractmethod
from threading import RLock

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
        **kwargs: Any,
    ) -> Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]:
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
            Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]:
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
