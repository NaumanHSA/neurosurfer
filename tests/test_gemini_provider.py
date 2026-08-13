"""Gemini, against a fake server speaking its real wire format.

The third shape, and the one that shares least with the other two: a different
role name for the assistant, the system prompt out of band, tool arguments
arriving already parsed, and tool results keyed by function name rather than by
call id. Each of those is a place a translation quietly goes wrong, so each has
a test.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from neurosurfer.llm.providers.gemini import (
    GeminiProvider,
    to_gemini_contents,
    to_gemini_tools,
)
from neurosurfer.llm.types import (
    Done,
    GenerationConfig,
    Message,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolSchema,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
)

TOOLS = [
    ToolSchema(
        name="read_file",
        description="read a file",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
]
CFG = GenerationConfig(max_tokens=256)


# ── a fake Gemini endpoint ──────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    chunks: list[dict] = []
    seen: list[dict] = []
    token_count = 42

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, "body": body})

        if ":countTokens" in self.path:
            payload = json.dumps({"totalTokens": type(self).token_count}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in type(self).chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        # A keep-alive and a terminator, both of which must be ignored.
        self.wfile.write(b": keep-alive\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a):
        pass


@pytest.fixture
def gemini():
    _Handler.chunks = []
    _Handler.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    yield GeminiProvider(model="gemini-2.5-flash", api_key="test", base_url=url), _Handler
    server.shutdown()


def _text_chunk(text, thought=False, finish=None):
    part = {"text": text}
    if thought:
        part["thought"] = True
    candidate = {"content": {"parts": [part]}}
    if finish:
        candidate["finishReason"] = finish
    return {"candidates": [candidate]}


async def _collect(provider, messages, tools=TOOLS):
    events, response = [], None
    async for event in provider.stream(messages, "system prompt", tools, CFG):
        if isinstance(event, Done):
            response = event.response
        else:
            events.append(event)
    return events, response


# ── streaming ───────────────────────────────────────────────────────────────


class TestStreaming:
    async def test_text_streams_and_assembles(self, gemini):
        provider, handler = gemini
        handler.chunks = [
            _text_chunk("Hello "),
            _text_chunk("world.", finish="STOP"),
        ]

        events, response = await _collect(provider, [Message.user_text("hi")])

        assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hello ", "world."]
        assert response.text() == "Hello world."
        assert response.stop_reason == "end_turn"

    async def test_fragments_merge_into_one_block(self, gemini):
        """One block per fragment would make a response hundreds of blocks long."""
        provider, handler = gemini
        handler.chunks = [_text_chunk("a"), _text_chunk("b"), _text_chunk("c")]

        _, response = await _collect(provider, [Message.user_text("hi")])

        assert len([b for b in response.content if isinstance(b, TextBlock)]) == 1

    async def test_thinking_is_separated_from_the_answer(self, gemini):
        """`thought: true` is reasoning — the same split every other provider makes."""
        provider, handler = gemini
        handler.chunks = [
            _text_chunk("weighing options", thought=True),
            _text_chunk("The answer.", finish="STOP"),
        ]

        events, response = await _collect(provider, [Message.user_text("hi")])

        assert [type(e).__name__ for e in events] == ["ThinkingDelta", "TextDelta"]
        assert isinstance(events[0], ThinkingDelta)
        assert response.text() == "The answer."
        assert any(isinstance(b, ThinkingBlock) for b in response.content)

    async def test_a_tool_call_is_native(self, gemini):
        provider, handler = gemini
        handler.chunks = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "read_file", "args": {"path": "x.py"}}}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        ]

        events, response = await _collect(provider, [Message.user_text("read x.py")])

        assert isinstance(events[0], ToolUseStart)
        assert events[0].name == "read_file"
        # Args arrive parsed; they are re-serialised so a consumer written
        # against the other providers sees the JSON-string shape it expects.
        assert isinstance(events[1], ToolUseArgsDelta)
        assert json.loads(events[1].partial_json) == {"path": "x.py"}

        [block] = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert block.input == {"path": "x.py"}
        assert response.stop_reason == "tool_use"

    async def test_parallel_tool_calls_get_distinct_ids(self, gemini):
        provider, handler = gemini
        handler.chunks = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "read_file", "args": {"path": "a"}}},
                                {"functionCall": {"name": "read_file", "args": {"path": "b"}}},
                            ]
                        }
                    }
                ]
            }
        ]

        _, response = await _collect(provider, [Message.user_text("read both")])

        ids = [b.id for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(ids) == 2 and len(set(ids)) == 2

    async def test_usage_includes_thinking_tokens(self, gemini):
        """`candidatesTokenCount` excludes thinking, which is still billed as
        output — dropping it would under-report the cost."""
        provider, handler = gemini
        handler.chunks = [
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 7,
                    "cachedContentTokenCount": 3,
                },
            }
        ]

        _, response = await _collect(provider, [Message.user_text("hi")])

        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 12
        assert response.usage.cache_read_input_tokens == 3

    @pytest.mark.parametrize(
        "reason,expected",
        [("STOP", "end_turn"), ("MAX_TOKENS", "max_tokens"), ("SAFETY", "refusal")],
    )
    async def test_stop_reasons_map_to_the_canonical_vocabulary(
        self, gemini, reason, expected
    ):
        provider, handler = gemini
        handler.chunks = [_text_chunk("x", finish=reason)]

        _, response = await _collect(provider, [Message.user_text("hi")])

        assert response.stop_reason == expected

    async def test_keepalives_and_done_markers_are_ignored(self, gemini):
        provider, handler = gemini
        handler.chunks = [_text_chunk("only this", finish="STOP")]

        _, response = await _collect(provider, [Message.user_text("hi")])

        assert response.text() == "only this"


# ── request shape ───────────────────────────────────────────────────────────


class TestRequestShape:
    async def test_the_system_prompt_is_out_of_band(self, gemini):
        """Gemini takes it as `systemInstruction`, not as a turn."""
        provider, handler = gemini
        handler.chunks = [_text_chunk("ok", finish="STOP")]

        await _collect(provider, [Message.user_text("hi")])

        body = handler.seen[0]["body"]
        assert body["systemInstruction"]["parts"][0]["text"] == "system prompt"
        assert all(c["role"] != "system" for c in body["contents"])

    async def test_max_tokens_is_forwarded(self, gemini):
        provider, handler = gemini
        handler.chunks = [_text_chunk("ok", finish="STOP")]

        await _collect(provider, [Message.user_text("hi")])

        assert handler.seen[0]["body"]["generationConfig"]["maxOutputTokens"] == 256


# ── message translation ─────────────────────────────────────────────────────


class TestMessageTranslation:
    def test_the_assistant_role_is_called_model(self):
        contents = to_gemini_contents(
            [Message.user_text("hi"), Message.assistant_text("hello")]
        )

        assert [c["role"] for c in contents] == ["user", "model"]

    def test_a_tool_result_is_keyed_by_function_name_not_call_id(self):
        """Gemini matches a `functionResponse` to its call by *name*, so the id
        has to be resolved back — otherwise the model sees a response to a
        function it never called."""
        messages = [
            Message.user_text("read it"),
            Message(
                role="assistant",
                content=[ToolUseBlock(id="call_abc", name="read_file", input={"path": "x"})],
            ),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="call_abc", content="file contents")],
            ),
        ]

        contents = to_gemini_contents(messages)

        response_part = contents[-1]["parts"][0]["functionResponse"]
        assert response_part["name"] == "read_file"
        assert response_part["response"] == {"result": "file contents"}

    def test_an_unmatched_tool_result_falls_back_to_its_id(self):
        """A worse name than the real one, and much better than dropping it."""
        contents = to_gemini_contents(
            [Message(role="user", content=[ToolResultBlock(tool_use_id="orphan", content="x")])]
        )

        assert contents[0]["parts"][0]["functionResponse"]["name"] == "orphan"

    def test_thinking_is_not_replayed(self):
        """Gemini will not accept it back, and sending it as ordinary text would
        feed the model its own private working notes as though they were answers."""
        messages = [
            Message(
                role="assistant",
                content=[ThinkingBlock(thinking="private"), TextBlock(text="public")],
            )
        ]

        [content] = to_gemini_contents(messages)

        assert content["parts"] == [{"text": "public"}]

    def test_an_empty_message_is_dropped(self):
        assert to_gemini_contents([Message(role="user", content=[TextBlock(text="")])]) == []


class TestToolTranslation:
    def test_unsupported_schema_keys_are_stripped(self):
        """Gemini's schema dialect is a subset of JSON Schema and errors on the rest."""
        [declaration] = to_gemini_tools(TOOLS)

        assert "additionalProperties" not in json.dumps(declaration)
        assert declaration["parameters"]["properties"]["path"]["type"] == "string"

    def test_a_tool_with_no_parameters_omits_the_key(self):
        """Gemini rejects a parameters object with no properties."""
        [declaration] = to_gemini_tools(
            [ToolSchema(name="ping", description="ping", input_schema={"type": "object"})]
        )

        assert "parameters" not in declaration


# ── token counting ──────────────────────────────────────────────────────────


class TestCountTokens:
    async def test_it_uses_the_real_endpoint(self, gemini):
        provider, handler = gemini
        handler.token_count = 123

        assert await provider.count_tokens([Message.user_text("hi")], "sys") == 123
        assert ":countTokens" in handler.seen[0]["path"]

    async def test_a_failure_degrades_to_an_estimate(self):
        """Counting is advisory — a failure must not fail a run that was only
        deciding whether to compact."""
        provider = GeminiProvider(api_key="test", base_url="http://127.0.0.1:9")

        assert await provider.count_tokens([Message.user_text("hello there")]) > 0


class TestConstruction:
    def test_a_missing_key_says_which_variables_it_looked_at(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            GeminiProvider()

    def test_it_reads_either_env_var(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "from-google-var")

        assert GeminiProvider().api_key == "from-google-var"
