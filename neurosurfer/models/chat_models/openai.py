# neurosurfer/models/openai_model.py
import logging
import uuid
import re
import os
from threading import Lock
from typing import Any, Generator, List, Dict, Union, Tuple, Optional
from datetime import datetime

from .base import BaseChatModel
from neurosurfer.server.schemas import ChatCompletionResponse, ChatCompletionChunk
from neurosurfer.config import config


class OpenAIModel(BaseChatModel):
    """
    An OpenAI/compatible chat client implementing BaseChatModel with Pydantic responses.
    
    Works with:
      • OpenAI Cloud (leave base_url=None, set a real api_key)
      • LM Studio local server (e.g., base_url="http://localhost:1234/v1", api_key="lm-studio")
      • vLLM OpenAI server (e.g., base_url="http://localhost:8000/v1")
      • Ollama OpenAI compat (e.g., base_url="http://localhost:11434/v1", api_key="ollama")
    """

    REASONING_BLOCKS: List[Tuple[str, str]] = [
        (r"<\|begin_of_thought\|>", r"<\|end_of_thought\|>"),
        (r"<think>", r"</think>"),
        (r"<analysis>", r"</analysis>"),
        (r"``````"),
    ]

    def __init__(
        self,
        model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None),
        timeout: Optional[float] = 120.0,
        stop_words: Optional[List[str]] = config.base_model.stop_words,
        strip_reasoning: bool = False,
        max_seq_length: int = config.base_model.max_seq_length,
        verbose: bool = config.base_model.verbose,
        logger: logging.Logger = logging.getLogger(),
        **kwargs: Any,
    ):
        super().__init__(max_seq_length=max_seq_length, stop_words=stop_words, verbose=verbose, logger=logger, **kwargs)
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.strip_reasoning = strip_reasoning
        self.tokenizer = None
        self.lock = Lock()
        self.client = None
        self.init_model()

    def init_model(self):
        try:
            from openai import OpenAI
        except Exception as e:
            raise ImportError(
                "Could not import OpenAI client. Please install with: pip install --upgrade openai"
            ) from e

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        if self.verbose:
            self.logger.info(
                f"[OpenAIModel] Initialized client for model={self.model_name} base_url={self.base_url or 'default'}"
            )
        
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"[OpenAIModel] Tokenizer not loaded: {e}")
            self.tokenizer = None

    def _format_messages(
        self,
        system_prompt: str,
        chat_history: List[dict],
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _common_args(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        allowed = {
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "n",
            "response_format",
            "seed",
        }
        passthrough = {k: v for k, v in extra.items() if k in allowed and v is not None}

        args: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            **passthrough,
        }
        if self.stop_words:
            args["stop"] = self.stop_words
        return args

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except:
                pass
        return len(text.split())  # Fallback approximation

    def _call(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Returns ChatCompletionResponse (Pydantic model)"""
        from openai import APIConnectionError, RateLimitError, APIStatusError

        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()
        
        with self.lock:
            messages = self._format_messages(system_prompt, chat_history, user_prompt)
            args = self._common_args(
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                extra=kwargs,
            )
            
            try:
                completion = self.client.chat.completions.create(timeout=self.timeout, **args)
                content = completion.choices[0].message.content or ""
                
                # Strip reasoning if enabled
                if self.strip_reasoning:
                    content = self.strip_reasoning_text(content)
                
                # Calculate token usage
                prompt_tokens = getattr(completion.usage, 'prompt_tokens', 0) if hasattr(completion, 'usage') else 0
                completion_tokens = getattr(completion.usage, 'completion_tokens', 0) if hasattr(completion, 'usage') else 0
                
                # Fallback to estimation if not provided
                if prompt_tokens == 0:
                    prompt_text = " ".join([m["content"] for m in messages])
                    prompt_tokens = self._estimate_tokens(prompt_text)
                if completion_tokens == 0:
                    completion_tokens = self._estimate_tokens(content)
                
                return self._final_nonstream_response(
                    call_id=self.call_id,
                    model=self.model_name,
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            
            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                err = f"[OpenAIModel] API error: {type(e).__name__}: {e}"
                if self.verbose:
                    self.logger.error(err)
                return self._final_nonstream_response(
                    call_id=self.call_id,
                    model=self.model_name,
                    content=err,
                    prompt_tokens=0,
                    completion_tokens=0,
                )

    def _stream(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Yields ChatCompletionChunk (Pydantic models)"""
        from openai import APIConnectionError, RateLimitError, APIStatusError

        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()
        
        with self.lock:
            messages = self._format_messages(system_prompt, chat_history, user_prompt)
            args = self._common_args(
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                extra=kwargs,
            )
            buffer = ""
            start_yield = True
            is_first_yield = True
            
            try:
                stream = self.client.chat.completions.create(stream=True, timeout=self.timeout, **args)
                for chunk in stream:
                    if self._stop_signal:
                        break
                    
                    for choice in chunk.choices:
                        piece: str = getattr(choice.delta, "content", None) or ""
                        if not piece:
                            continue
                        
                        buffer += piece
                        
                        # Client-side stop word check
                        if any(s in buffer for s in self.stop_words):
                            stream.close()
                            break
                        
                        # Strip reasoning blocks if enabled
                        if self.strip_reasoning:
                            for start_tag, end_tag in self.REASONING_BLOCKS:
                                if start_tag in piece:
                                    start_yield = False
                                if end_tag in piece:
                                    start_yield = True
                                    piece = ""
                        
                        if start_yield and piece:
                            # Remove leading whitespace from first yield
                            if is_first_yield:
                                piece = piece.lstrip()
                                is_first_yield = False
                            
                            yield self._delta_chunk(
                                call_id=self.call_id,
                                model=self.model_name,
                                content=piece,
                            )
                
                # Final stop chunk
                yield self._stop_chunk(
                    call_id=self.call_id,
                    model=self.model_name,
                    finish_reason="stop",
                )
            
            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                err = f"[OpenAIModel] API error: {type(e).__name__}: {e}"
                if self.verbose:
                    self.logger.error(err)
                yield self._delta_chunk(
                    call_id=self.call_id,
                    model=self.model_name,
                    content=err,
                )
                yield self._stop_chunk(
                    call_id=self.call_id,
                    model=self.model_name,
                    finish_reason="error",
                )

    def strip_reasoning_text(self, text: str) -> str:
        """Remove reasoning blocks from text"""
        for start, end in self.REASONING_BLOCKS:
            text = re.sub(start + r".*?" + end, "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    def stop_generation(self):
        """Set stop signal to halt generation"""
        self.set_stop_signal()

    def set_stop_words(self, stops: List[str]):
        """Update stop words list"""
        self.stop_words = stops or []
