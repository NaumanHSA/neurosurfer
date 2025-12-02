# neurosurfer/models/transformers_model.py
import os
import json
import uuid
import logging
import threading
import re
from typing import Any, Dict, Generator, List, Optional, Literal
from threading import RLock

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .base import BaseChatModel
from neurosurfer.server.schemas import ChatCompletionResponse, ChatCompletionChunk
from neurosurfer.config import config


class StopSignalCriteria(StoppingCriteria):
    def __init__(self, stop_fn):
        super().__init__()
        self.stop_fn = stop_fn

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_fn()


class TransformersModel(BaseChatModel):
    """
    HF Transformers local model - returns Pydantic models.
    """

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = config.base_model.max_seq_length,
        load_in_4bit: bool = config.base_model.load_in_4bit,
        enable_thinking: bool = config.base_model.enable_thinking,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        stop_words: Optional[List[str]] = config.base_model.stop_words,
        verbose: bool = config.base_model.verbose,
        logger: logging.Logger = logging.getLogger(),
        **kwargs,
    ):
        super().__init__(
            max_seq_length=max_seq_length,
            enable_thinking=enable_thinking,
            stop_words=stop_words,
            verbose=verbose,
            logger=logger,
            **kwargs,
        )
        self.model_name = model_name
        self.lock = RLock()
        self.call_id: Optional[str] = None
        self.generation_thread: Optional[threading.Thread] = None
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort

        # Choose dtype/device sensibly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.init_model(load_in_4bit, **kwargs)

    # ---------- public controls ----------
    def stop_generation(self):
        self.logger.info("[TransformersModel] Stop signal set.")
        self.set_stop_signal()
        if getattr(self, "generation_thread", None) and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=0.5)

    def set_stop_words(self, stops: List[str]):
        self.stop_words = stops or []

    # ---------- init ----------
    def init_model(self, load_in_4bit: bool = False, **kwargs: Any):
        self.logger.info("Initializing Transformers model.")
        model_args: Dict[str, Any] = {
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }
        is_prequantized = self._is_model_already_quantized(self.model_name)

        if load_in_4bit and not is_prequantized:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_args["quantization_config"] = bnb_config
            except Exception as e:
                self.logger.warning(f"Could not enable 4-bit quantization: {e}")
        elif load_in_4bit and is_prequantized:
            self.logger.warning("Model is already quantized. Ignoring load_in_4bit=True.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_args)
        if self.device == "cpu":
            self.model.to(self.dtype)
        self.model.eval()
        self.logger.info("Transformers model initialized successfully.")

    def _is_model_already_quantized(self, model_path_or_id: str) -> bool:
        """Check if model is already quantized."""
        cfg = os.path.join(model_path_or_id, "config.json")
        if os.path.exists(cfg):
            with open(cfg, "r") as f:
                try:
                    config_json = json.load(f)
                    return "quantization_config" in config_json
                except Exception:
                    return False

        try:
            from huggingface_hub import snapshot_download
            tmp_dir = snapshot_download(repo_id=model_path_or_id, allow_patterns=["config.json"])
            cfg2 = os.path.join(tmp_dir, "config.json")
            if os.path.exists(cfg2):
                with open(cfg2, "r") as f:
                    config_json = json.load(f)
                    return "quantization_config" in config_json
        except Exception:
            pass
        return False

    # ---------- core (non-stream) - Returns Pydantic Model ----------
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
        Returns ChatCompletionResponse (Pydantic model).
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()

        # Format prompt using BaseChatModel (chat_template aware)
        messages = self._build_messages(system_prompt, chat_history, user_prompt)
        # 2) Tokenized inputs via chat_template-aware helper
        inputs = self._apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        ).to("cuda")
        # if self.device == "cuda":
        #     inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        prompt_tokens = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9 if not self.enable_thinking else 0.95,
            **kwargs,
        )

        generated_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(generated_tokens)

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if not self.enable_thinking:
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        return self._final_nonstream_response(
            call_id=self.call_id,
            model=self.model_name,
            content=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ---------- core (stream) - Yields Pydantic Models ----------
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
        Yields ChatCompletionChunk (Pydantic models).
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

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9 if not self.enable_thinking else 0.95,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
        )
        gen_kwargs.update(kwargs)

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
