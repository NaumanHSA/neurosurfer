"""Shared provider resolution for the real-LLM integration tests.

By default the real-LLM suites target a local LM Studio server
(``qwen/qwen3.5-9b`` at ``http://localhost:1234/v1``) and auto-skip when it is
not reachable — CI stays hermetic. The model/endpoint are overridable via env so
the same tests can run against a hosted model (e.g. OpenAI ``gpt-4o-mini``):

    NEUROSURFER_TEST_MODEL      model id            (default ``qwen/qwen3.5-9b``)
    NEUROSURFER_TEST_BASE_URL   OpenAI-compat URL   (default LM Studio; the values
                                ``""`` / ``openai`` / ``https://api.openai.com/v1``
                                select the hosted OpenAI provider)
    NEUROSURFER_TEST_API_KEY    key (falls back to ``OPENAI_API_KEY``)

Example — run the architect suite on a hosted model:

    NEUROSURFER_TEST_BASE_URL=openai NEUROSURFER_TEST_MODEL=gpt-4o-mini \\
        conda run -n LLMs python -m pytest tests/test_architect_agent_llm.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

_HOSTED_OPENAI = {"", "openai", "https://api.openai.com/v1", "https://api.openai.com"}


def _load_dotenv_key(name: str) -> str | None:
    """Best-effort read of a single key from the repo-root .env (no dependency)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


def _get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _load_dotenv_key(name) or default


MODEL: str = _get("NEUROSURFER_TEST_MODEL", "qwen/qwen3.5-9b")  # type: ignore[assignment]
BASE_URL: str = _get("NEUROSURFER_TEST_BASE_URL", "http://localhost:1234/v1")  # type: ignore[assignment]
_API_KEY: str | None = _get("NEUROSURFER_TEST_API_KEY") or _get("OPENAI_API_KEY")


def is_hosted() -> bool:
    return (BASE_URL or "").rstrip("/") in {u.rstrip("/") for u in _HOSTED_OPENAI}


def provider_ready() -> bool:
    """True when the configured model/endpoint is usable (else the suite skips)."""
    if is_hosted():
        return bool(_API_KEY and _API_KEY != "not-needed")
    try:
        import httpx

        resp = httpx.get(f"{BASE_URL}/models", timeout=3.0)
        return MODEL in [m.get("id") for m in resp.json().get("data", [])]
    except Exception:  # noqa: BLE001
        return False


def skip_reason() -> str:
    if is_hosted():
        return f"hosted model {MODEL} needs OPENAI_API_KEY / NEUROSURFER_TEST_API_KEY"
    return f"LM Studio with {MODEL} not reachable at {BASE_URL}"


def make_provider():
    """Build the provider for the configured model (hosted OpenAI or OpenAI-compat)."""
    import warnings

    from neurosurfer.llm.providers.openai import OpenAICompatProvider, OpenAIProvider

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_hosted():
            return OpenAIProvider(api_key=_API_KEY, model=MODEL)
        return OpenAICompatProvider(
            base_url=BASE_URL, api_key=_API_KEY or "not-needed",
            model=MODEL, context_window=32768,
        )
