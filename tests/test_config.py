from pathlib import Path

from neurosurfer.config import DEFAULT_ANTHROPIC_MODEL, load_config


def test_defaults_anthropic(monkeypatch):
    for k in ("LLM_PROVIDER", "MODEL", "ANTHROPIC_API_KEY", "CONTEXT_WINDOW"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_config(env_file=Path("/nonexistent/.env"))
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == DEFAULT_ANTHROPIC_MODEL
    assert cfg.llm.is_anthropic and not cfg.llm.is_openai


def test_openai_selection(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MODEL", "qwen2.5-coder")
    monkeypatch.setenv("CONTEXT_WINDOW", "32768")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    cfg = load_config(env_file=Path("/nonexistent/.env"))
    assert cfg.llm.is_openai
    assert cfg.llm.model == "qwen2.5-coder"
    assert cfg.llm.context_window == 32768
    assert cfg.llm.openai_base_url == "http://localhost:8000/v1"


def test_unknown_provider_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "banana")
    cfg = load_config(env_file=Path("/nonexistent/.env"))
    assert cfg.llm.provider == "anthropic"


def test_redacted_masks_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    cfg = load_config(env_file=Path("/nonexistent/.env"))
    red = cfg.redacted()
    assert "secret" not in str(red["anthropic_api_key"])
