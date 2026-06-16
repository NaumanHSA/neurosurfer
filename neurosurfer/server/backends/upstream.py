from __future__ import annotations
import json
from typing import AsyncIterator, Optional, Tuple
import httpx
from .base import Backend
from ..errors import OpenAIHTTPError

class UpstreamBackend(Backend):
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str = "",
        models_mode: str = "proxy",
        static_models: Optional[dict] = None,
        timeout: float = 120.0,
    ):
        self._name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models_mode = models_mode
        self.static_models = static_models
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=None))

    @property
    def name(self) -> str:
        return self._name

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def list_models(self) -> dict:
        if self.models_mode == "static" and self.static_models is not None:
            return self.static_models
        r = await self._client.get(f"{self.base_url}/models", headers=self._headers())
        if r.status_code >= 400:
            raise OpenAIHTTPError(r.status_code, f"Upstream /models failed: {r.text}")
        return r.json()

    async def chat_completions(self, req: dict, *, request_id: str) -> Tuple[bool, object]:
        url = f"{self.base_url}/chat/completions"
        stream = bool(req.get("stream"))

        if not stream:
            r = await self._client.post(url, headers=self._headers(), json=req)
            if r.status_code >= 400:
                try:
                    payload = r.json()
                    msg = (payload.get("error") or {}).get("message") or r.text
                except Exception:
                    msg = r.text
                raise OpenAIHTTPError(r.status_code, msg)
            return False, r.json()

        async def gen() -> AsyncIterator[dict]:
            async with self._client.stream("POST", url, headers=self._headers(), json=req) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise OpenAIHTTPError(resp.status_code, body.decode("utf-8", "ignore"))
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except Exception:
                            continue

        return True, gen()

    async def aclose(self):
        await self._client.aclose()
