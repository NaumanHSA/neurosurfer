# neurosurfer/models/unsloth_model.py
import os
import logging
import uuid
import threading
import re
import inspect
from threading import RLock
from typing import Any, Generator, List, Optional, Literal
from neurosurfer.runtime.checks import require
require("unsloth", "Unsloth Framwork", "pip install unsloth")

from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

from .base import BaseChatModel
from neurosurfer.server.schemas import ChatCompletionResponse, ChatCompletionChunk
from neurosurfer.config import config

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


class StopSignalCriteria(StoppingCriteria):
    """Custom stopping criteria that checks a stop signal function."""
    def __init__(self, stop_fn):
        super().__init__()
        self.stop_fn = stop_fn

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        return self.stop_fn()


class UnslothModel(BaseChatModel):
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
        load_in_8bit: bool = False,
        full_finetuning: bool = False,
        enable_thinking: bool = config.base_model.enable_thinking,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        add_special_tokens: bool = False,
        stop_words: Optional[List[str]] = config.base_model.stop_words,
        verbose: bool = config.base_model.verbose,
        logger: logging.Logger = logging.getLogger(),
        **kwargs: Any,
    ):
        super().__init__(
            max_seq_length=max_seq_length,
            stop_words=stop_words,
            enable_thinking=enable_thinking,
            verbose=verbose,
            logger=logger,
            **kwargs,
        )
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.full_finetuning = full_finetuning
        self.add_special_tokens = add_special_tokens
        self.call_id: Optional[str] = None
        self.lock = RLock()
        self.stop_words = stop_words or []
        self.generation_thread: Optional[threading.Thread] = None

        self.model = None
        self.tokenizer = None

        self.init_model(**kwargs)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    @staticmethod
    def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
        """
        Keep only kwargs accepted by fn. This lets us pass arbitrary **kwargs
        into UnslothModel(...) and silently drop unsupported ones for
        FastLanguageModel.from_pretrained.
        """
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}

    def init_model(self, **kwargs: Any):
        """Initialize Unsloth model with specified configuration."""
        self.logger.info(f"Initializing Unsloth model: {self.model_name}")
        try:
            filtered_kwargs = self._filter_kwargs_for_callable(
                FastLanguageModel.from_pretrained,
                kwargs,
            )
            ignored = set(kwargs.keys()) - set(filtered_kwargs.keys())
            if ignored:
                self.logger.info(
                    f"[UnslothModel] Ignoring unsupported model init kwargs: {ignored}"
                )

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                full_finetuning=self.full_finetuning,
                fast_inference=False,
                **filtered_kwargs,
            )
            FastLanguageModel.for_inference(self.model)
            self.model.eval()
            self.logger.info("Unsloth model initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Unsloth model: {e}")
            raise Exception(f"Unsloth model couldn't load properly: {e}")

    # ------------------------------------------------------------------
    # Non-streaming call
    # ------------------------------------------------------------------
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
        Synchronous generation that returns ChatCompletionResponse (Pydantic model).
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()

        # 1) Build messages via BaseChatModel
        messages = self._build_messages(system_prompt, chat_history, user_prompt)

        # 2) Tokenized inputs via chat_template-aware helper
        inputs = self._apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        ).to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]
        # 3) Streamer & stopping criteria
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        stopping_criteria = StoppingCriteriaList([
            StopSignalCriteria(lambda: self._stop_signal),
        ])

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

        with self.lock:
            self.generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=gen_kwargs,
                daemon=True,
            )
            self.generation_thread.start()

        # 4) Consume stream into a single response string
        self.logger.debug("[UnslothModel] Entered Call --- Consuming Stream now ...")
        response = ""
        for piece in self._transformers_consume_stream(streamer):
            response += piece

        # Optional thinking suppression if your prompts wrap it in <think>...</think>
        if not self.enable_thinking:
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # 5) Token counting (approximate fallback)
        try:
            completion_tokens = len(
                self.tokenizer.encode(
                    response,
                    add_special_tokens=self.add_special_tokens,
                )
            )
        except Exception as e:
            self.logger.warning(f"Failed to encode response for token counting: {e}")
            completion_tokens = int(len(response.split()) * 1.3)

        return self._final_nonstream_response(
            call_id=self.call_id,
            model=self.model_name,
            content=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ------------------------------------------------------------------
    # Streaming call
    # ------------------------------------------------------------------
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
        Streaming generation that yields ChatCompletionChunk (Pydantic models).
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()

        # 1) Build messages
        messages = self._build_messages(system_prompt, chat_history, user_prompt)

        # 2) Tokenized inputs via chat_template-aware helper
        inputs = self._apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        ).to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        stopping_criteria = StoppingCriteriaList([
            StopSignalCriteria(lambda: self._stop_signal),
        ])

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

        with self.lock:
            self.generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs=gen_kwargs,
                daemon=True,
            )
            self.generation_thread.start()

        for piece in self._transformers_consume_stream(streamer):
            yield self._delta_chunk(
                call_id=self.call_id,
                model=self.model_name,
                content=piece,
            )

        yield self._stop_chunk(
            call_id=self.call_id,
            model=self.model_name,
            finish_reason="stop",
        )

    # ------------------------------------------------------------------
    # Stop words & interruption API
    # ------------------------------------------------------------------
    def set_stop_words(self, stops: List[str]):
        """Update stop words list."""
        self.stop_words = stops or []
        self.logger.info(f"[UnslothModel] Stop words updated: {self.stop_words}")

    def stop_generation(self):
        """
        Signal the model to stop generation immediately.
        Thread-safe operation.
        """
        self.logger.info("[UnslothModel] Stop signal set.")
        self.set_stop_signal()

        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=0.5)
