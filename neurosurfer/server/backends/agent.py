from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional, Tuple
import anyio
from .base import Backend
from ..schemas.openai import ChatCompletionResponse, ChatCompletionChoice
from ..streaming.openai_chunks import chunk_role, chunk_text, chunk_end
from ..errors import OpenAIHTTPError

def _default_result_to_text(result: Any) -> str:
    if result is None:
        return ""
    if hasattr(result, "model_dump"):
        try:
            d = result.model_dump()
            if isinstance(d, dict):
                if "final" in d and isinstance(d["final"], dict):
                    final = d["final"]
                    if len(final) == 1:
                        return str(next(iter(final.values())))
                    return json.dumps(final, ensure_ascii=False, indent=2, default=str)
                return json.dumps(d, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
    if isinstance(result, dict):
        for k in ("final_answer", "answer", "output", "result", "text"):
            if k in result:
                return str(result[k])
        if len(result) == 1:
            return str(next(iter(result.values())))
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    return str(result)

@dataclass
class AgentSpec:
    agent: Any
    model_id: str
    description: str = "Neurosurfer agent"
    owned_by: str = "neurosurfer"
    max_model_len: int = 8192
    run_fn: Optional[Callable[[Any, str, list], Any]] = None
    result_to_text: Callable[[Any], str] = _default_result_to_text

class AgentBackend(Backend):
    def __init__(self, spec: AgentSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.model_id

    async def list_models(self) -> dict:
        now = int(time.time())
        perm = [{
            "id": f"modelperm-{uuid.uuid4().hex[:16]}",
            "object": "model_permission",
            "created": now,
            "allow_sampling": True,
            "allow_logprobs": True,
            "allow_view": True,
            "is_blocking": False,
        }]
        return {"object": "list", "data": [{
            "id": self.spec.model_id,
            "object": "model",
            "created": now,
            "owned_by": self.spec.owned_by,
            "max_model_len": self.spec.max_model_len,
            "permission": perm,
        }]}

    def _invoke_blocking(self, user_query: str, chat_history: list) -> Any:
        agent = self.spec.agent
        if self.spec.run_fn is not None:
            return self.spec.run_fn(agent, user_query, chat_history)
        if hasattr(agent, "run"):
            try:
                return agent.run(inputs={"query": user_query})
            except TypeError:
                pass
            try:
                return agent.run(user_query=user_query, chat_history=chat_history)
            except TypeError:
                pass
            try:
                return agent.run(user_query)
            except TypeError:
                pass
        raise RuntimeError("Agent does not support a known invocation pattern")

    async def chat_completions(self, req: dict, *, request_id: str) -> Tuple[bool, object]:
        model = req.get("model") or self.spec.model_id
        messages = req.get("messages") or []
        user_query = ""
        chat_history = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                user_query = content if isinstance(content, str) else json.dumps(content, default=str)
            if role in ("user", "assistant"):
                chat_history.append({"role": role, "content": content})

        if not user_query:
            raise OpenAIHTTPError(400, "No user message found")

        result = await anyio.to_thread.run_sync(self._invoke_blocking, user_query, chat_history)
        text = self.spec.result_to_text(result)

        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if bool(req.get("stream")):
            async def gen() -> AsyncIterator[dict]:
                yield chunk_role(id=completion_id, created=created, model=model)
                if text:
                    yield chunk_text(id=completion_id, created=created, model=model, text=text)
                yield chunk_end(id=completion_id, created=created, model=model, finish_reason="stop")
            return True, gen()

        resp = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": text}, finish_reason="stop")],
        ).model_dump()
        return False, resp
