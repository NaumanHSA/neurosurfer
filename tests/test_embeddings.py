"""The embeddings layer: spec parsing, the two failure modes, and the wire format.

This was 69 lines with a three-branch `if` and one backend, and it is what every
retrieval path depends on. The tests that matter most here are the ones about
*which* failure a caller gets — returning `None` for an expired API key is how
"your credentials lapsed" becomes "search quietly got worse".
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from neurosurfer.embeddings import (
    EmbedderUnavailable,
    EmbeddingError,
    OpenAICompatEmbedder,
    get_embedder,
    parse_spec,
)

# ── spec parsing ────────────────────────────────────────────────────────────


class TestParseSpec:
    def test_a_bare_model_name_is_sentence_transformers(self):
        """What it meant before this package existed. `RAGAgentConfig` ships
        `intfloat/e5-small-v2` today and must keep resolving."""
        assert parse_spec("intfloat/e5-small-v2") == ("local", {"model": "intfloat/e5-small-v2"})

    def test_bare_backend_names(self):
        assert parse_spec("local") == ("local", {})
        assert parse_spec("openai") == ("openai", {})

    def test_a_prefixed_model(self):
        assert parse_spec("openai:text-embedding-3-large") == (
            "openai",
            {"model": "text-embedding-3-large"},
        )

    def test_openai_compat_carries_a_base_url(self):
        assert parse_spec("openai-compat:nomic-embed@http://localhost:1234/v1") == (
            "openai-compat",
            {"model": "nomic-embed", "base_url": "http://localhost:1234/v1"},
        )

    def test_a_model_name_containing_a_slash_survives_the_at_split(self):
        got = parse_spec("openai-compat:org/model-v2@https://host:8443/v1")
        assert got[1] == {"model": "org/model-v2", "base_url": "https://host:8443/v1"}

    def test_openai_compat_without_a_base_url_is_a_clear_error(self):
        with pytest.raises(ValueError, match="needs a base URL"):
            parse_spec("openai-compat:some-model")


# ── the two failure modes ───────────────────────────────────────────────────


class TestOffIsNotAFailure:
    @pytest.mark.parametrize("spec", ["none", "off", "bm25", "null", "lexical", "", None])
    def test_switching_embeddings_off_returns_none(self, spec):
        assert get_embedder(spec) is None


class TestNotConfiguredDegrades:
    def test_a_missing_api_key_returns_none(self, monkeypatch):
        """Not configured. Falling back to lexical search is a fair answer."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        assert get_embedder("openai:text-embedding-3-small") is None


class TestConfiguredAndBrokenRaises:
    def test_a_malformed_spec_raises_rather_than_silently_degrading(self):
        with pytest.raises(ValueError, match="needs a base URL"):
            get_embedder("openai-compat:no-url-here")

    def test_degrade_restores_the_old_never_raises_behaviour(self):
        """For a caller that genuinely wants to limp — a background re-index."""
        assert get_embedder("openai-compat:no-url-here", degrade=True) is None


# ── the wire format, against a real HTTP server ─────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    """Minimal `/v1/embeddings`. Returns rows out of order on purpose."""

    fail_times = 0
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)

        if type(self).fail_times > 0:
            type(self).fail_times -= 1
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"{}")
            return

        n = len(body["input"])
        # Deliberately reversed: the spec permits any order, and a client that
        # trusts position instead of `index` mis-pairs text to vector.
        rows = [
            {"index": i, "embedding": [float(i), 1.0]} for i in reversed(range(n))
        ]
        payload = json.dumps({"data": rows}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # silence the default stderr logging
        pass


@pytest.fixture
def fake_endpoint():
    _Handler.fail_times = 0
    _Handler.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", _Handler
    server.shutdown()


class TestOpenAICompat:
    def test_it_embeds(self, fake_endpoint):
        url, _ = fake_endpoint
        emb = OpenAICompatEmbedder(model="m", base_url=url)

        assert emb.embed(["a", "b", "c"]) == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]

    def test_results_are_ordered_by_index_not_by_arrival(self, fake_endpoint):
        """The server returns them reversed; `index` is what pairs them back."""
        url, _ = fake_endpoint
        vecs = OpenAICompatEmbedder(model="m", base_url=url).embed(["a", "b"])

        assert vecs[0][0] == 0.0 and vecs[1][0] == 1.0

    def test_batching_splits_the_request(self, fake_endpoint):
        url, handler = fake_endpoint
        emb = OpenAICompatEmbedder(model="m", base_url=url, max_batch=2)

        emb.embed(["a", "b", "c", "d", "e"])

        assert [len(r["input"]) for r in handler.seen] == [2, 2, 1]

    def test_dimensions_are_learned_from_the_first_call(self, fake_endpoint):
        url, _ = fake_endpoint
        emb = OpenAICompatEmbedder(model="m", base_url=url)
        assert emb.dimensions is None

        emb.embed(["a"])
        assert emb.dimensions == 2

    def test_a_503_is_retried(self, fake_endpoint):
        url, handler = fake_endpoint
        handler.fail_times = 2
        emb = OpenAICompatEmbedder(model="m", base_url=url, max_attempts=4)

        assert emb.embed(["a"]) == [[0.0, 1.0]]
        assert len(handler.seen) == 3

    def test_giving_up_raises_embedding_error_not_none(self, fake_endpoint):
        """The whole point: a broken endpoint is loud."""
        url, handler = fake_endpoint
        handler.fail_times = 99
        emb = OpenAICompatEmbedder(model="m", base_url=url, max_attempts=2)

        with pytest.raises(EmbeddingError, match="failed to embed"):
            emb.embed(["a"])

    def test_an_endpoint_that_is_not_an_embeddings_api_says_so(self, monkeypatch):
        emb = OpenAICompatEmbedder(model="m", base_url="http://x/v1")
        with pytest.raises(EmbeddingError, match="OpenAI-compatible"):
            emb._parse({"choices": []})

    def test_no_texts_makes_no_request(self, fake_endpoint):
        url, handler = fake_endpoint
        assert OpenAICompatEmbedder(model="m", base_url=url).embed([]) == []
        assert handler.seen == []

    def test_a_missing_base_url_is_unavailable_not_a_crash(self):
        with pytest.raises(EmbedderUnavailable):
            OpenAICompatEmbedder(model="m", base_url="")


def test_get_embedder_builds_a_compat_backend(fake_endpoint):
    url, _ = fake_endpoint
    emb = get_embedder(f"openai-compat:m@{url}")

    assert emb is not None
    assert emb.embed(["a"]) == [[0.0, 1.0]]
