"""Tests for neurosurfer.server (N1 — OpenAI-compatible gateway).

These tests cover:
- Schema models (serialisation / extra-field passthrough)
- SSE / chunk helpers
- Hook lifecycle (before_chat / after_chat / stream_chunk)
- OpenAIHTTPError formatting
- ModelRouter routing logic
- AgentBackend with sync, async-coroutine, and async-generator agents
- NeurosurferServer construction (no real HTTP; just checks FastAPI app is built)
- CLI serve command imports without error
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# ── import guard: skip the whole module if fastapi is not installed ──────────
pytest.importorskip("fastapi")

from neurosurfer.app.server.backends.agent import AgentBackend, AgentSpec, _default_result_to_text
from neurosurfer.app.server.backends.base import Backend
from neurosurfer.app.server.errors import OpenAIHTTPError
from neurosurfer.app.server.gateway import NeurosurferServer
from neurosurfer.app.server.hooks.base import Hook, HookContext
from neurosurfer.app.server.hooks.builtin import StripReasoningHook, SystemPromptInjectorHook
from neurosurfer.app.server.registry import ModelRouter, RouteTarget
from neurosurfer.app.server.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
)
from neurosurfer.app.server.streaming.openai_chunks import chunk_end, chunk_role, chunk_text
from neurosurfer.app.server.streaming.sse import sse_data, sse_done

# ── Schemas ───────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_completion_request_serialises(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="hello")],
            stream=True,
        )
        d = req.model_dump()
        assert d["model"] == "gpt-4"
        assert d["stream"] is True


# ── SSE helpers ───────────────────────────────────────────────────────────────

class TestSSE:
    def test_sse_data_dict(self):
        b = sse_data({"a": 1})
        assert b.startswith(b"data: ")
        assert b.endswith(b"\n\n")
        payload = json.loads(b[6:].strip())
        assert payload == {"a": 1}

    def test_sse_done(self):
        assert sse_done() == b"data: [DONE]\n\n"


# ── Chunk helpers ─────────────────────────────────────────────────────────────

class TestChunks:
    def test_chunk_role(self):
        c = chunk_role(id="id1", created=1, model="m")
        assert c["object"] == "chat.completion.chunk"
        assert c["choices"][0]["delta"]["role"] == "assistant"

    def test_chunk_text(self):
        c = chunk_text(id="id1", created=1, model="m", text="hello")
        assert c["choices"][0]["delta"]["content"] == "hello"

    def test_chunk_end(self):
        c = chunk_end(id="id1", created=1, model="m")
        assert c["choices"][0]["finish_reason"] == "stop"
        assert c["choices"][0]["delta"].get("content") is None


# ── Errors ────────────────────────────────────────────────────────────────────

class TestErrors:
    def test_to_openai_error(self):
        e = OpenAIHTTPError(404, "not found", error_type="invalid_request_error")
        d = e.to_openai_error()
        assert d["error"]["message"] == "not found"
        assert d["error"]["type"] == "invalid_request_error"


# ── Hooks ─────────────────────────────────────────────────────────────────────

class TestHooks:
    def _ctx(self) -> HookContext:
        return HookContext(request_id="r1", model="m", user=None, client_ip=None)

    @pytest.mark.asyncio
    async def test_base_hook_passthrough(self):
        hk = Hook()
        ctx = self._ctx()
        req = {"model": "x"}
        assert await hk.before_chat(ctx, req) is req
        resp = {"id": "1"}
        assert await hk.after_chat(ctx, resp) is resp
        chunk = {"choices": []}
        assert await hk.stream_chunk(ctx, chunk) is chunk

    @pytest.mark.asyncio
    async def test_strip_reasoning_hook_after_chat(self):
        hk = StripReasoningHook()
        ctx = self._ctx()
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "<think>thinking</think>answer"}}]
        }
        out = await hk.after_chat(ctx, resp)
        assert out["choices"][0]["message"]["content"] == "answer"

    @pytest.mark.asyncio
    async def test_system_prompt_injector(self):
        hk = SystemPromptInjectorHook("Be helpful.")
        ctx = self._ctx()
        req = {"messages": [{"role": "user", "content": "hello"}]}
        out = await hk.before_chat(ctx, req)
        assert out["messages"][0]["role"] == "system"
        assert out["messages"][0]["content"] == "Be helpful."

    @pytest.mark.asyncio
    async def test_system_prompt_not_injected_if_present(self):
        hk = SystemPromptInjectorHook("Be helpful.")
        ctx = self._ctx()
        req = {
            "messages": [
                {"role": "system", "content": "existing system"},
                {"role": "user", "content": "hello"},
            ]
        }
        out = await hk.before_chat(ctx, req)
        assert out["messages"][0]["content"] == "existing system"


# ── ModelRouter ───────────────────────────────────────────────────────────────

class TestModelRouter:
    def _dummy_backend(self, name: str) -> Backend:
        b = MagicMock(spec=Backend)
        b.name = name
        return b

    def test_resolve_registered(self):
        router = ModelRouter()
        b = self._dummy_backend("b1")
        router.register_model("m1", RouteTarget(backend=b))
        target = router.resolve("m1")
        assert target.backend is b

    def test_resolve_default_backend(self):
        router = ModelRouter()
        b = self._dummy_backend("default")
        router.set_default_backend(b)
        target = router.resolve("unknown-model")
        assert target.backend is b
        assert target.upstream_model == "unknown-model"

    def test_resolve_unknown_raises(self):
        router = ModelRouter()
        with pytest.raises(KeyError):
            router.resolve("not-registered")

    def test_all_models(self):
        router = ModelRouter()
        b = self._dummy_backend("b")
        router.register_model("m1", RouteTarget(backend=b))
        router.register_model("m2", RouteTarget(backend=b))
        assert set(router.all_models()) == {"m1", "m2"}


# ── AgentBackend ──────────────────────────────────────────────────────────────

def _spec(agent, **kwargs) -> AgentSpec:
    return AgentSpec(agent=agent, model_id="test-agent", **kwargs)


class TestAgentBackend:
    @pytest.mark.asyncio
    async def test_list_models(self):
        backend = AgentBackend(_spec(object()))
        result = await backend.list_models()
        assert result["data"][0]["id"] == "test-agent"

    # ── run_fn path ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_run_fn_sync(self):
        def my_run(agent, query, history):
            return f"echo:{query}"

        agent = object()
        backend = AgentBackend(_spec(agent, run_fn=my_run))
        req = {"model": "test-agent", "messages": [{"role": "user", "content": "hello"}]}
        is_stream, result = await backend.chat_completions(req, request_id="r1")
        assert not is_stream
        assert result["choices"][0]["message"]["content"] == "echo:hello"

    # ── async generator agent (native streaming) ─────────────────────────────

    @pytest.mark.asyncio
    async def test_async_gen_agent_streaming(self):
        @dataclass
        class TextDelta:
            text: str

        class GenAgent:
            async def run(self, prompt: str):
                yield TextDelta("chunk1")
                yield TextDelta("chunk2")

        backend = AgentBackend(_spec(GenAgent()))
        req = {
            "model": "test-agent",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        is_stream, result = await backend.chat_completions(req, request_id="r5")
        assert is_stream
        chunks = [c async for c in result]
        # role chunk + 2 text chunks + end chunk
        assert len(chunks) == 4
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[1]["choices"][0]["delta"]["content"] == "chunk1"
        assert chunks[2]["choices"][0]["delta"]["content"] == "chunk2"
        assert chunks[3]["choices"][0]["finish_reason"] == "stop"

    # ── token usage in the response ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_usage_in_blocking_response(self):
        from neurosurfer.llm.types import Usage

        class Result:
            usage = Usage(input_tokens=7, output_tokens=3)

        backend = AgentBackend(
            _spec(object(), run_fn=lambda a, q, h: Result(), result_to_text=lambda r: "ok")
        )
        req = {"model": "test-agent", "messages": [{"role": "user", "content": "hi"}]}
        _, result = await backend.chat_completions(req, request_id="u1")
        assert result["usage"] == {
            "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10
        }

    @pytest.mark.asyncio
    async def test_usage_chunk_when_stream_options_include_usage(self):
        from neurosurfer.llm.types import Usage

        @dataclass
        class TextDelta:
            text: str

        class UsageGenAgent:
            def __init__(self):
                self.usage = Usage()

            async def run(self, prompt: str):
                yield TextDelta("hi")
                self.usage = Usage(input_tokens=10, output_tokens=5)  # accrues during run

        backend = AgentBackend(_spec(UsageGenAgent()))
        req = {
            "model": "test-agent",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        _, result = await backend.chat_completions(req, request_id="u2")
        chunks = [c async for c in result]
        # role + text + end + usage
        assert chunks[-1]["usage"] == {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
        }
        assert chunks[-1]["choices"] == []
        # without include_usage, no usage chunk is emitted
        req2 = {**req, "stream_options": {}}
        _, result2 = await backend.chat_completions(req2, request_id="u3")
        chunks2 = [c async for c in result2]
        assert all(c.get("usage") is None for c in chunks2)

    # ── error: no user message ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_no_user_message_raises(self):
        backend = AgentBackend(_spec(object(), run_fn=lambda a, q, h: ""))
        req = {"model": "test-agent", "messages": [{"role": "system", "content": "sys"}]}
        with pytest.raises(OpenAIHTTPError) as exc_info:
            await backend.chat_completions(req, request_id="r7")
        assert exc_info.value.status_code == 400


# ── result_to_text ────────────────────────────────────────────────────────────

class TestResultToText:
    def test_dict_with_answer_key(self):
        assert _default_result_to_text({"answer": "42"}) == "42"


# ── NeurosurferServer construction ────────────────────────────────────────────

class TestNeurosurferServer:
    def test_builds_fastapi_app(self):
        server = NeurosurferServer(host="127.0.0.1", port=9999, enable_docs=False)
        from fastapi import FastAPI

        assert isinstance(server.app, FastAPI)

    def test_settings_overrides(self):
        server = NeurosurferServer(port=7777, log_level="debug")
        assert server.settings.port == 7777
        assert server.settings.log_level == "debug"

    def test_register_agent_creates_backend(self):
        @dataclass
        class FakeEvent:
            text: str

        class FakeAgent:
            async def run(self, prompt: str):
                yield FakeEvent("ok")

        server = NeurosurferServer()
        server.register_agent(FakeAgent(), model_id="fake-agent")
        target = server.router.resolve("fake-agent")
        assert isinstance(target.backend, AgentBackend)
        assert target.backend.spec.model_id == "fake-agent"

    def test_add_hook(self):
        server = NeurosurferServer()
        hk = StripReasoningHook()
        server.add_hook(hk)
        assert hk in server.hooks


# ── CLI serve command ─────────────────────────────────────────────────────────

class TestServeCLI:
    def test_serve_parser_registered(self):

        from neurosurfer.app.cli import build_parser

        parser = build_parser()
        # Parse serve --help should not raise; just get the namespace
        args = parser.parse_args(["serve", "--port", "9000"])
        assert args.command == "serve"
        assert args.port == 9000
