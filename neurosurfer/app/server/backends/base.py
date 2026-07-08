from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def list_models(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    async def chat_completions(self, req: dict, *, request_id: str) -> tuple[bool, object]:
        """Return ``(is_stream, result)``.

        ``is_stream=False`` → result is a JSON-serialisable dict.
        ``is_stream=True``  → result is an ``AsyncIterator[dict]`` of OpenAI chunks.
        """
        raise NotImplementedError
