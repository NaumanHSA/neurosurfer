from pathlib import Path

from neurosurfer.config import DEFAULT_ANTHROPIC_MODEL, load_config
from neurosurfer.config.base import load_dotenv


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


def test_dotenv_strips_inline_comments_but_keeps_quoted_hashes(tmp_path, monkeypatch):
    for k in ("A", "B", "C", "D"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "A=https://host.example.com     # trailing comment\n"
        'B="value # not a comment"\n'
        "C=has#nospace-kept\n"
        "D=plain\n"
    )
    load_dotenv(env)
    import os

    assert os.environ["A"] == "https://host.example.com"   # inline comment stripped
    assert os.environ["B"] == "value # not a comment"       # quoted '#' preserved
    assert os.environ["C"] == "has#nospace-kept"            # '#' with no space kept
    assert os.environ["D"] == "plain"
