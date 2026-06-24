"""Tests for neurosurfer.cache — hit/miss/expiry for all backends."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from neurosurfer.cache import (
    CachedEmbedder,
    CachedProvider,
    DiskResponseCache,
    InMemoryResponseCache,
    get_response_cache,
)
from neurosurfer.cache.base import CacheKey
from neurosurfer.cache.provider import _make_key
from neurosurfer.llm.types import (
    CanonicalResponse,
    GenerationConfig,
    Message,
    TextBlock,
    Usage,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _response(text: str = "hello") -> CanonicalResponse:
    return CanonicalResponse(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
        model="test-model",
    )


def _key(tag: str = "k1") -> CacheKey:
    return CacheKey(key=f"{'a' * 60}{tag[:4]}")


def _messages() -> list[Message]:
    return [Message.user_text("What is 2+2?")]


def _config(**kw) -> GenerationConfig:
    return GenerationConfig(**{**{"max_tokens": 512, "temperature": 0.5}, **kw})


# ─────────────────────────────────────────────────────────────────────────────
# InMemoryResponseCache
# ─────────────────────────────────────────────────────────────────────────────

class TestInMemoryResponseCache:
    def test_miss_returns_none(self):
        c = InMemoryResponseCache()
        assert c.get(_key()) is None

    def test_hit_returns_response(self):
        c = InMemoryResponseCache()
        r = _response()
        c.set(_key(), r)
        got = c.get(_key())
        assert got is not None
        assert got.text() == "hello"

    def test_hit_increments_hits(self):
        c = InMemoryResponseCache()
        c.set(_key(), _response())
        c.get(_key())
        c.get(_key())
        entry = c._store[_key().key]
        assert entry.hits == 2

    def test_expiry_returns_none(self):
        c = InMemoryResponseCache(ttl=0.01)
        c.set(_key(), _response())
        time.sleep(0.05)
        assert c.get(_key()) is None

    def test_no_expiry_when_ttl_none(self):
        c = InMemoryResponseCache(ttl=None)
        c.set(_key(), _response())
        # Manually wind back creation time
        c._store[_key().key].created_at = time.time() - 99999
        assert c.get(_key()) is not None

    def test_lru_eviction(self):
        c = InMemoryResponseCache(maxsize=2)
        k1, k2, k3 = _key("k1"), _key("k2"), _key("k3")
        c.set(k1, _response("r1"))
        c.set(k2, _response("r2"))
        c.set(k3, _response("r3"))  # evicts k1
        assert c.get(k1) is None
        assert c.get(k2) is not None
        assert c.get(k3) is not None

    def test_lru_access_updates_order(self):
        c = InMemoryResponseCache(maxsize=2)
        k1, k2, k3 = _key("k1"), _key("k2"), _key("k3")
        c.set(k1, _response("r1"))
        c.set(k2, _response("r2"))
        c.get(k1)            # k1 is now MRU
        c.set(k3, _response("r3"))  # should evict k2, not k1
        assert c.get(k1) is not None
        assert c.get(k2) is None
        assert c.get(k3) is not None

    def test_clear(self):
        c = InMemoryResponseCache()
        c.set(_key(), _response())
        c.clear()
        assert c.size() == 0

    def test_size(self):
        c = InMemoryResponseCache()
        assert c.size() == 0
        c.set(_key("k1"), _response())
        c.set(_key("k2"), _response())
        assert c.size() == 2

    def test_size_excludes_expired(self):
        c = InMemoryResponseCache(ttl=0.01)
        c.set(_key("k1"), _response())
        c.set(_key("k2"), _response())
        time.sleep(0.05)
        assert c.size() == 0


# ─────────────────────────────────────────────────────────────────────────────
# DiskResponseCache
# ─────────────────────────────────────────────────────────────────────────────

class TestDiskResponseCache:
    def test_miss_returns_none(self, tmp_path):
        c = DiskResponseCache(tmp_path / "cache")
        assert c.get(_key()) is None

    def test_hit_roundtrips_response(self, tmp_path):
        c = DiskResponseCache(tmp_path / "cache")
        r = _response("disk test")
        c.set(_key(), r)
        got = c.get(_key())
        assert got is not None
        assert got.text() == "disk test"
        assert got.stop_reason == "end_turn"
        assert got.usage.input_tokens == 10

    def test_expiry_returns_none(self, tmp_path):
        c = DiskResponseCache(tmp_path / "cache", ttl=0.01)
        c.set(_key(), _response())
        time.sleep(0.05)
        assert c.get(_key()) is None

    def test_expired_file_is_deleted(self, tmp_path):
        c = DiskResponseCache(tmp_path / "cache", ttl=0.01)
        c.set(_key(), _response())
        path = c._path(_key())
        assert path.exists()
        time.sleep(0.05)
        c.get(_key())  # triggers deletion
        assert not path.exists()

    def test_no_expiry_when_ttl_none(self, tmp_path):
        c = DiskResponseCache(tmp_path / "cache", ttl=None)
        c.set(_key(), _response())
        # Overwrite created_at to ancient time
        import json
        path = c._path(_key())
        data = json.loads(path.read_text())
        data["created_at"] = 0.0
        path.write_text(json.dumps(data))
        assert c.get(_key()) is not None

    def test_clear(self, tmp_path):
        c = DiskResponseCache(tmp_path / "cache")
        c.set(_key("k1"), _response())
        c.set(_key("k2"), _response())
        c.clear()
        assert c.size() == 0

    def test_size(self, tmp_path):
        c = DiskResponseCache(tmp_path / "cache")
        assert c.size() == 0
        c.set(_key("k1"), _response())
        c.set(_key("k2"), _response())
        assert c.size() == 2

    def test_corrupt_file_returns_none(self, tmp_path):
        c = DiskResponseCache(tmp_path / "cache")
        c.set(_key(), _response())
        c._path(_key()).write_text("not json {{{{")
        assert c.get(_key()) is None


# ─────────────────────────────────────────────────────────────────────────────
# CachedProvider
# ─────────────────────────────────────────────────────────────────────────────

def _make_provider(response_text: str = "answer") -> MagicMock:
    provider = MagicMock()
    provider.model = "test-model"
    provider.capabilities = MagicMock()
    provider.complete = AsyncMock(return_value=_response(response_text))
    provider.count_tokens = AsyncMock(return_value=42)
    return provider


class TestCachedProvider:
    def test_cache_none_is_passthrough(self):
        inner = _make_provider()
        p = CachedProvider(inner, cache=None)
        asyncio.run(p.complete(_messages(), None, [], _config()))
        asyncio.run(p.complete(_messages(), None, [], _config()))
        assert inner.complete.call_count == 2

    def test_cache_hit_skips_provider(self):
        inner = _make_provider()
        cache = InMemoryResponseCache()
        p = CachedProvider(inner, cache=cache)
        r1 = asyncio.run(p.complete(_messages(), None, [], _config()))
        r2 = asyncio.run(p.complete(_messages(), None, [], _config()))
        assert inner.complete.call_count == 1
        assert r1.text() == r2.text() == "answer"

    def test_different_args_produce_different_keys(self):
        inner = _make_provider()
        cache = InMemoryResponseCache()
        p = CachedProvider(inner, cache=cache)
        asyncio.run(p.complete(_messages(), "sys1", [], _config()))
        asyncio.run(p.complete(_messages(), "sys2", [], _config()))
        assert inner.complete.call_count == 2
        assert cache.size() == 2

    def test_different_temperature_different_keys(self):
        inner = _make_provider()
        cache = InMemoryResponseCache()
        p = CachedProvider(inner, cache=cache)
        asyncio.run(p.complete(_messages(), None, [], _config(temperature=0.1)))
        asyncio.run(p.complete(_messages(), None, [], _config(temperature=0.9)))
        assert inner.complete.call_count == 2

    def test_count_tokens_delegates(self):
        inner = _make_provider()
        p = CachedProvider(inner, cache=None)
        result = asyncio.run(p.count_tokens(_messages(), None, []))
        assert result == 42

    def test_stream_delegates_without_caching(self):
        inner = _make_provider()
        inner.stream = MagicMock(return_value=iter([]))
        cache = InMemoryResponseCache()
        p = CachedProvider(inner, cache=cache)
        p.stream(_messages(), None, [], _config())
        inner.stream.assert_called_once()
        assert cache.size() == 0

    def test_expiry_causes_re_call(self):
        inner = _make_provider("v1")
        cache = InMemoryResponseCache(ttl=0.01)
        p = CachedProvider(inner, cache=cache)
        asyncio.run(p.complete(_messages(), None, [], _config()))
        time.sleep(0.05)
        inner.complete = AsyncMock(return_value=_response("v2"))
        r2 = asyncio.run(p.complete(_messages(), None, [], _config()))
        assert r2.text() == "v2"
        assert inner.complete.call_count == 1  # second call re-hit provider


# ─────────────────────────────────────────────────────────────────────────────
# CachedEmbedder
# ─────────────────────────────────────────────────────────────────────────────

class _FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(i)] for i in range(len(texts))]


class TestCachedEmbedder:
    def test_miss_calls_inner(self):
        inner = _FakeEmbedder()
        c = CachedEmbedder(inner)
        c.embed(["a", "b"])
        assert inner.calls == 1

    def test_hit_skips_inner(self):
        inner = _FakeEmbedder()
        c = CachedEmbedder(inner)
        r1 = c.embed(["a", "b"])
        r2 = c.embed(["a", "b"])
        assert inner.calls == 1
        assert r1 == r2

    def test_different_texts_different_keys(self):
        inner = _FakeEmbedder()
        c = CachedEmbedder(inner)
        c.embed(["hello"])
        c.embed(["world"])
        assert inner.calls == 2
        assert c.size() == 2

    def test_lru_eviction(self):
        inner = _FakeEmbedder()
        c = CachedEmbedder(inner, maxsize=2)
        c.embed(["a"])
        c.embed(["b"])
        c.embed(["c"])  # evicts "a"
        assert c.size() == 2
        # "a" is gone — next call hits inner
        pre = inner.calls
        c.embed(["a"])
        assert inner.calls == pre + 1

    def test_maxsize_zero_is_passthrough(self):
        inner = _FakeEmbedder()
        c = CachedEmbedder(inner, maxsize=0)
        c.embed(["x"])
        c.embed(["x"])
        assert inner.calls == 2

    def test_clear(self):
        inner = _FakeEmbedder()
        c = CachedEmbedder(inner)
        c.embed(["a"])
        c.clear()
        assert c.size() == 0

    def test_empty_texts_passthrough(self):
        inner = _FakeEmbedder()
        c = CachedEmbedder(inner)
        result = c.embed([])
        assert result == []
        assert inner.calls == 1


# ─────────────────────────────────────────────────────────────────────────────
# get_response_cache factory
# ─────────────────────────────────────────────────────────────────────────────

class TestGetResponseCache:
    def test_none_returns_none(self):
        assert get_response_cache(None) is None
        assert get_response_cache("off") is None
        assert get_response_cache("disabled") is None

    def test_memory_backend(self):
        c = get_response_cache("memory", maxsize=10, ttl=60)
        assert isinstance(c, InMemoryResponseCache)
        assert c.maxsize == 10
        assert c.ttl == 60

    def test_disk_backend(self, tmp_path):
        c = get_response_cache("disk", directory=tmp_path / "c", ttl=None)
        assert isinstance(c, DiskResponseCache)
        assert c.ttl is None

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown cache backend"):
            get_response_cache("redis")
