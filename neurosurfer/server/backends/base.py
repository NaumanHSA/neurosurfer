from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

class Backend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def list_models(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    async def chat_completions(self, req: dict, *, request_id: str) -> Tuple[bool, object]:
        raise NotImplementedError
