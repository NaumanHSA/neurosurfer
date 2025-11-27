# neurosurfer/models/transformers_model.py
import os
import json
import uuid
import logging
import threading
import re
from typing import Any, Dict, Generator, List, Optional, Tuple
from threading import RLock
from jinja2.exceptions import TemplateError

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
    HF Transformers local model - now returns Pydantic models
    """

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = config.base_model.max_seq_length,
        load_in_4bit: bool = config.base_model.load_in_4bit,
        enable_thinking: bool = config.base_model.enable_thinking,
        stop_words: Optional[List[str]] = config.base_model.stop_words,
        verbose: bool = config.base_model.verbose,
        logger: logging.Logger = logging.getLogger(),
        **kwargs,
    ):
        super().__init__(max_seq_length=max_seq_length, enable_thinking=enable_thinking, stop_words=stop_words, verbose=verbose, logger=logger, **kwargs)
        self.model_name = model_name
        self.lock = RLock()
        self.call_id: Optional[str] = None
        self.generation_thread: Optional[threading.Thread] = None

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
    def init_model(self, load_in_4bit=False, **kwargs):
        self.logger.info("Initializing Transformers model.")
        model_args = {
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
        """Check if model is already quantized"""
        cfg = os.path.join(model_path_or_id, "config.json")
        if os.path.exists(cfg):
            with open(cfg, "r") as f:
                try:
                    config = json.load(f)
                    return "quantization_config" in config
                except Exception:
                    return False

        try:
            from huggingface_hub import snapshot_download
            tmp_dir = snapshot_download(repo_id=model_path_or_id, allow_patterns=["config.json"])
            cfg2 = os.path.join(tmp_dir, "config.json")
            if os.path.exists(cfg2):
                with open(cfg2, "r") as f:
                    config = json.load(f)
                    return "quantization_config" in config
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
        Returns ChatCompletionResponse (Pydantic model)
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()

        # Format prompt
        prompt = self._format_messages(system_prompt, chat_history, user_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        prompt_tokens = inputs["input_ids"].shape[1]

        # Run generation synchronously
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9 if not self.enable_thinking else 0.95,
            **kwargs,
        )

        # Slice off the prompt tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        completion_tokens = len(generated_tokens)

        # Decode only the new tokens
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Remove <think>...</think> blocks if thinking is disabled
        if not self.enable_thinking:
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # Return Pydantic model
        return self._final_nonstream_response(
            call_id=self.call_id,
            model=self.model_name,
            content=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
    
    # ---------- core (stream) - Yields Pydantic Models ----------
    def _stream(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any
    ) -> Generator[ChatCompletionChunk, None, None]:
        """
        Yields ChatCompletionChunk (Pydantic models)
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()

        prompt = self._format_messages(system_prompt, chat_history, user_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        stopping_criteria = StoppingCriteriaList([StopSignalCriteria(lambda: self._stop_signal)])

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
            self.generation_thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
            self.generation_thread.start()

        # Stream out Pydantic chunks
        for piece in self._transformers_consume_stream(streamer):
            yield self._delta_chunk(call_id=self.call_id, model=self.model_name, content=piece)
        # Final stop chunk
        yield self._stop_chunk(call_id=self.call_id, model=self.model_name, finish_reason="stop")
    
    # ---------- formatting ----------
    def _format_messages(
        self,
        system_prompt: str,
        chat_history: List[dict],
        user_prompt: str
    ) -> str:
        # Qwen thinking toggle
        if "qwen3" in self.model_name.lower() and not self.enable_thinking:
            system_prompt = (system_prompt or "") + "/nothink"

        # Normalize conversation for strict templates
        system_prompt, normalized_history = self._normalize_for_chat_template(
            system_prompt=system_prompt,
            chat_history=chat_history or [],
            user_prompt=user_prompt,
        )

        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.extend(normalized_history)

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TemplateError as e:
            # Log and fall back to a simple text format
            self.logger.warning(
                "Chat template failed (%s). Falling back to plain prompt formatting.",
                e,
            )
            return self._fallback_plain_prompt(system_prompt, normalized_history)

    def _fallback_plain_prompt(
        self,
        system_prompt: Optional[str],
        messages: List[Dict[str, str]],
    ) -> str:
        parts = []
        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}\n")

        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"[{role}]\n{content}\n")

        # End with an assistant cue
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)