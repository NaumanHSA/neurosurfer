# neurosurfer/models/unsloth_model.py
import os
import logging
import uuid
import threading
import re
from threading import Lock, RLock
from typing import Any, Generator, List, Optional, Tuple, Union
from neurosurfer.runtime.checks import require
require("unsloth", "Unsloth Framwork", "pip install unsloth")

from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

from .base import BaseModel
from neurosurfer.server.schemas import ChatCompletionResponse, ChatCompletionChunk
from neurosurfer.config import config


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

class StopSignalCriteria(StoppingCriteria):
    """Custom stopping criteria that checks a stop signal function"""
    def __init__(self, stop_fn):
        super().__init__()
        self.stop_fn = stop_fn

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_fn()


class UnslothModel(BaseModel):
    """
    Unsloth FastLanguageModel wrapper with Pydantic response models.
    
    Features:
    - Returns ChatCompletionResponse for non-streaming
    - Yields ChatCompletionChunk for streaming
    - Thread-safe stop signal
    - Stop words support (truncates before stop sequence)
    - Optional thinking mode with <think> tag suppression
    - Token usage tracking
    """

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = config.base_model.max_seq_length,
        load_in_4bit: bool = config.base_model.load_in_4bit,
        load_in_8bit: bool = True,
        full_finetuning: bool = False,
        enable_thinking: bool = config.base_model.enable_thinking,
        stop_words: Optional[List[str]] = config.base_model.stop_words,
        verbose: bool = config.base_model.verbose,
        logger: logging.Logger = logging.getLogger(),
        **kwargs,
    ):
        super().__init__(max_seq_length=max_seq_length, stop_words=stop_words, enable_thinking=enable_thinking, verbose=verbose, logger=logger, **kwargs)
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.full_finetuning = full_finetuning
        self.call_id: Optional[str] = None
        self.lock = RLock()  # Use RLock for nested lock support
        self.stop_words = stop_words or []
        self.generation_thread: Optional[threading.Thread] = None
        self.init_model(**kwargs)

    # ---------- Initialization ----------
    def init_model(self, **kwargs):
        """Initialize Unsloth model with specified configuration"""
        self.logger.info(f"Initializing Unsloth model: {self.model_name}")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                full_finetuning=self.full_finetuning,
                fast_inference=False,
                **kwargs,
            )
            FastLanguageModel.for_inference(self.model)
            self.model.eval()
            self.logger.info("Unsloth model initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Unsloth model: {e}")
            raise Exception(f"Unsloth model couldn't load properly: {e}")

    # ---------- Non-Streaming Call - Returns Pydantic Model ----------
    def _call(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """
        Synchronous generation that returns ChatCompletionResponse (Pydantic model)
        
        Returns:
            ChatCompletionResponse with full message and token usage
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()

        # Format the prompt
        prompt_text = self._format_messages(system_prompt, chat_history, user_prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]

        # Setup streamer (we still use streaming internally for stop word support)
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        # Stopping criteria
        stopping_criteria = StoppingCriteriaList([StopSignalCriteria(lambda: self._stop_signal)])

        # Generation parameters
        top_p = 0.95 if self.enable_thinking else 0.9
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 20,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria,
            **kwargs,
        }

        # Run generation in background thread
        with self.lock:
            self.generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=gen_kwargs,
                daemon=True
            )
            self.generation_thread.start()

        # Consume the stream and aggregate response
        print("[UnslothModel] Entered Call --- Consuming Stream now ...")
        # Stream out Pydantic chunks
        response = ""
        for piece in self._transformers_consume_stream(streamer):
            response += piece
        if not self.enable_thinking:
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # Calculate token count (approximate for completion)
        try:
            completion_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
        except Exception as e:
            self.logger.warning(f"Failed to encode response for token counting: {e}")
            # Fallback: estimate tokens based on character count (rough approximation)
            completion_tokens = len(response.split()) * 1.3  # Rough estimate: 1.3 tokens per word

        # Return Pydantic model
        return self._final_nonstream_response(
            call_id=self.call_id,
            model=self.model_name,
            content=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(completion_tokens)
        )

    # ---------- Streaming Call - Yields Pydantic Models ----------
    def _stream(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """
        Streaming generation that yields ChatCompletionChunk (Pydantic models)
        
        Yields:
            ChatCompletionChunk objects with incremental delta content
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()

        # Format the prompt
        prompt_text = self._format_messages(system_prompt, chat_history, user_prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Stopping criteria
        stopping_criteria = StoppingCriteriaList([
            StopSignalCriteria(lambda: self._stop_signal)
        ])

        # Generation parameters
        top_p = 0.95 if self.enable_thinking else 0.9
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 20,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria,
            **kwargs,
        }

        # Run generation in background thread
        with self.lock:
            self.generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=gen_kwargs,
                daemon=True
            )
            self.generation_thread.start()

        # Stream out Pydantic chunks
        for piece in self._transformers_consume_stream(streamer):
            yield self._delta_chunk(
                call_id=self.call_id, 
                model=self.model_name, 
                content=piece
            )
        
        # Final stop chunk
        yield self._stop_chunk(
            call_id=self.call_id, 
            model=self.model_name, 
            finish_reason="stop"
        )

    # ---------- Format Messages ----------
    def _format_messages(
        self,
        system_prompt: str,
        chat_history: List[dict],
        user_prompt: str
    ) -> str:
        """
        Format messages using tokenizer's chat template.
        
        For Qwen models with thinking: appends "/nothink" to system prompt when disabled.
        """
        # Qwen thinking toggle
        if "qwen" in self.model_name.lower() and not self.enable_thinking:
            system_prompt = (system_prompt or "") + "/nothink"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(chat_history or [])
        messages.append({"role": "user", "content": user_prompt})

        # Check if tokenizer supports enable_thinking parameter
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def set_stop_words(self, stops: List[str]):
        """Update stop words list"""
        self.stop_words = stops or []
        self.logger.info(f"[UnslothModel] Stop words updated: {self.stop_words}")

    def stop_generation(self):
        """
        Signal the model to stop generation immediately.
        Thread-safe operation.
        """
        self.logger.info("[UnslothModel] Stop signal set.")
        self.reset_stop_signal()
        
        # Try to join the generation thread
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=0.5)
