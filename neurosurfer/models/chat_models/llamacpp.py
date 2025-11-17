# neurosurfer/models/llamacpp_model.py
import os
from typing import List, Optional, Any, Generator
import logging
import uuid
from threading import RLock
import threading
from neurosurfer.runtime.checks import require

from .base import BaseChatModel
from neurosurfer.server.schemas import ChatCompletionResponse, ChatCompletionChunk
from neurosurfer.config import config

class LlamaCppModel(BaseChatModel):
    """
    llama.cpp model wrapper with Pydantic responses.
    
    Features:
    - Supports both local GGUF files and HuggingFace repos
    - Streaming and non-streaming completions
    - Thread-safe generation
    - Token usage tracking
    - Stop signal support
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        n_ctx: int = config.base_model.max_seq_length,
        n_threads: int = 4,
        main_gpu: int = 0,
        n_gpu_layers: int = -1,
        stop_words: Optional[List[str]] = config.base_model.stop_words,
        verbose: bool = config.base_model.verbose,
        logger: logging.Logger = logging.getLogger(),
        **kwargs,
    ):
        super().__init__(max_seq_length=n_ctx, verbose=verbose, logger=logger, **kwargs)
        self.repo_id = repo_id
        self.filename = filename
        self.model_path = model_path
        self.call_id: Optional[str] = None
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.main_gpu = main_gpu
        self.n_gpu_layers = n_gpu_layers
        self.stop_words = stop_words or []
        self.lock = RLock()
        self.model = None
        self.model_name = None
        self.generation_active = False
        self.init_model(**kwargs)

    def init_model(self, **kwargs):
        """Initialize llama.cpp model from local file or HuggingFace repo"""
        require("llama_cpp", "Llama CPP Framwork", "pip install llama-cpp-python")
        from llama_cpp import Llama
        self.logger.info("[LlamaCppModel] Initializing llama.cpp model.")
        try:
            # Load from local GGUF file
            if self.model_path is not None and os.path.splitext(self.model_path)[-1] == ".gguf":
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
                self.model_name = os.path.basename(self.model_path)
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    main_gpu=self.main_gpu,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                    **kwargs,
                )
                self.logger.info(f"[LlamaCppModel] Loaded local model: {self.model_name}")
            
            # Load from HuggingFace
            elif self.repo_id and self.filename:
                self.model_name = f"{self.repo_id}/{self.filename}"
                self.model = Llama.from_pretrained(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    main_gpu=self.main_gpu,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                    **kwargs,
                )
                self.logger.info(f"[LlamaCppModel] Loaded HuggingFace model: {self.model_name}")
            
            else:
                raise ValueError(
                    "Either model_path (for local .gguf) or repo_id+filename (for HuggingFace) must be provided"
                )
        
        except Exception as e:
            self.logger.error(f"[LlamaCppModel] Failed to load model: {e}")
            raise Exception(f"llama.cpp model couldn't load properly: {e}")

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
        Non-streaming call that returns ChatCompletionResponse (Pydantic model)
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()
        
        with self.lock:
            self.generation_active = True
            
            try:
                # Format messages
                messages = self._format_messages(system_prompt, chat_history, user_prompt)
                # Prepare kwargs for llama.cpp
                llama_kwargs = self._prepare_generation_kwargs(
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    stream=False,
                    **kwargs
                )
                # Generate response
                response = self.model.create_chat_completion(messages=messages, **llama_kwargs)
                
                # Extract content and usage
                content = response["choices"][0]["message"]["content"]
                
                # Extract token usage
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                return self._final_nonstream_response(
                    call_id=self.call_id,
                    model=self.model_name,
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            
            except Exception as e:
                self.logger.error(f"[LlamaCppModel] Generation error: {e}")
                return self._final_nonstream_response(
                    call_id=self.call_id,
                    model=self.model_name,
                    content=f"[Error: {str(e)}]",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            
            finally:
                self.generation_active = False

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
        Streaming generator that yields ChatCompletionChunk (Pydantic models)
        """
        self.call_id = str(uuid.uuid4())
        self.reset_stop_signal()
        
        with self.lock:
            self.generation_active = True
            
            try:
                # Format messages
                messages = self._format_messages(system_prompt, chat_history, user_prompt)
                
                # Prepare kwargs for llama.cpp
                llama_kwargs = self._prepare_generation_kwargs(
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    stream=True,
                    **kwargs
                )
                
                # Stream generation
                rolling_buffer = ""
                
                for chunk in self.model.create_chat_completion(
                    messages=messages,
                    **llama_kwargs
                ):
                    # Check stop signal
                    if self._stop_signal:
                        self.logger.info("[LlamaCppModel] Stop signal detected.")
                        break
                    
                    # Extract delta content
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    
                    if not content:
                        continue
                    
                    rolling_buffer += content
                    
                    # Check stop words
                    hit, cutoff = self._find_first_stop(rolling_buffer, self.stop_words)
                    if hit is not None:
                        # Emit text before stop word
                        to_emit = rolling_buffer[:cutoff]
                        if to_emit:
                            yield self._delta_chunk(
                                call_id=self.call_id,
                                model=self.model_name,
                                content=to_emit
                            )
                        break
                    
                    # Emit the chunk
                    yield self._delta_chunk(
                        call_id=self.call_id,
                        model=self.model_name,
                        content=content
                    )
                    
                    # Keep buffer size manageable
                    if len(rolling_buffer) > 1000:
                        rolling_buffer = rolling_buffer[-500:]
            
            except Exception as e:
                self.logger.error(f"[LlamaCppModel] Streaming error: {e}")
                yield self._delta_chunk(
                    call_id=self.call_id,
                    model=self.model_name,
                    content=f"[Error: {str(e)}]"
                )
            
            finally:
                self.generation_active = False
            
        # Final stop chunk (OUTSIDE the with block and loop)
        yield self._stop_chunk(
            call_id=self.call_id,
            model=self.model_name,
            finish_reason="stop"
        )

    def _format_messages(
        self,
        system_prompt: str,
        chat_history: List[dict],
        user_prompt: str
    ) -> List[dict]:
        """Format messages for llama.cpp"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add chat history
        if chat_history:
            messages.extend(chat_history)
        
        # Add user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        return messages

    def _prepare_generation_kwargs(
        self,
        temperature: float,
        max_new_tokens: int,
        stream: bool,
        **kwargs
    ) -> dict:
        """Prepare kwargs for llama.cpp generation"""
        # llama.cpp uses max_tokens, not max_new_tokens
        generation_kwargs = {
            "stream": stream,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        
        # Add stop words if provided
        if self.stop_words:
            generation_kwargs["stop"] = self.stop_words
        
        # Pass through allowed kwargs
        allowed_params = {
            "top_p", "top_k", "repeat_penalty", "frequency_penalty",
            "presence_penalty", "mirostat_mode", "mirostat_tau",
            "mirostat_eta", "tfs_z", "typical_p", "seed"
        }
        
        for key, value in kwargs.items():
            if key in allowed_params and value is not None:
                generation_kwargs[key] = value
        
        return generation_kwargs

    def stop_generation(self):
        """Signal to stop the current generation"""
        self.logger.info("[LlamaCppModel] Stop signal set.")
        self.set_stop_signal()

    def set_stop_words(self, stops: List[str]):
        """Update stop words list"""
        self.stop_words = stops or []
        if self.verbose:
            self.logger.info(f"[LlamaCppModel] Stop words updated: {self.stop_words}")
