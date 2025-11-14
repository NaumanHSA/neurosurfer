from __future__ import annotations
import asyncio, time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class ModelProfile:
    name: str   
    device: str = "cpu"
    max_concurrency: int = 1
    supports_batching: bool = False
    reserved_mem_mb: int = 0
    max_tokens_per_req: int = 4096

@dataclass
class _Bucket:
    sem: asyncio.Semaphore
    profile: ModelProfile
    last_oom_at: float = 0.0
    shrink_factor: float = 1.0

class ModelPool:
    """
    Token/slot-based concurrency control.
    - Each model has a semaphore (slots).
    - On OOM, we record timestamp and shrink allowed tokens for a while.
    """
    def __init__(self):
        self._buckets: Dict[str, _Bucket] = {}

    @classmethod
    def from_llms(cls, llms: List[Any], default_concurrency: int = 1) -> "ModelPool":
        pool = cls()
        for llm in llms:
            name = getattr(llm, "model_name", "unknown")
            # crude heuristic: if local HF or llama.cpp, keep concurrency = 1 by default
            local = any(k in name.lower() for k in ["llama", "mistral", "qwen", "phi", "hf", "transformer"])
            maxc = 1 if local else default_concurrency
            profile = ModelProfile(name=name, max_concurrency=maxc)
            pool.register_model(name, profile)
        return pool

    def register_model(self, model_name: str, profile: ModelProfile):
        self._buckets[model_name] = _Bucket(asyncio.Semaphore(profile.max_concurrency), profile)

    def bucket(self, model_name: str) -> _Bucket:
        if model_name not in self._buckets:
            # default bucket
            self.register_model(model_name, ModelProfile(name=model_name))
        return self._buckets[model_name]

    async def acquire(self, model_name: str):
        return await self.bucket(model_name).sem.acquire()

    def release(self, model_name: str):
        self.bucket(model_name).sem.release()

    def notify_oom(self, model_name: str):
        b = self.bucket(model_name)
        b.last_oom_at = time.time()
        b.shrink_factor = max(0.5 * b.shrink_factor, 0.25)  # progressively shrink

    def recommend_max_new_tokens(self, model_name: str, requested: int) -> int:
        b = self.bucket(model_name)
        return max(64, int(requested * b.shrink_factor))
