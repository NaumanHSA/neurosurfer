"""Bedrock — the id scheme and the client, which is all that differs.

**Honest scope.** `boto3` is not installed here and there are no AWS
credentials, so this does not exercise a live Bedrock call. What it does check
is everything that is Bedrock-specific and not the shared Anthropic path: the
model-id translation both ways, that the shared translation is genuinely
inherited rather than copied, the local token estimate, and that a missing
dependency says which extra to install.
"""

from __future__ import annotations

import pytest

from neurosurfer.llm.providers.anthropic import AnthropicProvider
from neurosurfer.llm.providers.bedrock import (
    BedrockProvider,
    from_bedrock_model_id,
    to_bedrock_model_id,
)
from neurosurfer.llm.types import Message


class TestModelIds:
    def test_a_first_party_id_gains_the_prefix(self):
        """A bare id sent to Bedrock is a 400; adding it here means the same
        config string works against either provider."""
        assert to_bedrock_model_id("claude-opus-5") == "anthropic.claude-opus-5"

    def test_it_is_idempotent(self):
        assert to_bedrock_model_id("anthropic.claude-opus-5") == "anthropic.claude-opus-5"

    def test_whitespace_is_tolerated(self):
        assert to_bedrock_model_id("  claude-haiku-4-5 ") == "anthropic.claude-haiku-4-5"

    def test_the_bare_id_can_be_recovered(self):
        """Capabilities and prices are keyed on first-party names, so the
        prefix has to come back off for those lookups."""
        assert from_bedrock_model_id("anthropic.claude-opus-5") == "claude-opus-5"
        assert from_bedrock_model_id("claude-opus-5") == "claude-opus-5"

    def test_a_round_trip_is_stable(self):
        assert from_bedrock_model_id(to_bedrock_model_id("claude-sonnet-5")) == "claude-sonnet-5"


class TestItInheritsRatherThanCopies:
    """The reason this is a subclass: the translation is already correct.

    If these ever stop being the same function, there are two implementations
    of the Messages API translation in this repository and one of them is
    quietly wrong.
    """

    @pytest.mark.parametrize(
        "method", ["stream", "_build_kwargs", "complete"]
    )
    def test_the_shared_path_is_not_reimplemented(self, method):
        assert getattr(BedrockProvider, method) is getattr(AnthropicProvider, method)

    def test_only_construction_and_counting_are_overridden(self):
        """`__init__` is deliberately in scope here — it is one of the two
        things Bedrock legitimately changes, so filtering all dunders out would
        make the assertion weaker than it reads."""
        own = {
            name
            for name, value in vars(BedrockProvider).items()
            if callable(value) and (not name.startswith("__") or name == "__init__")
        }

        assert own == {"__init__", "count_tokens"}


class TestCountTokens:
    async def test_it_estimates_locally(self):
        """Bedrock has no count_tokens endpoint — the parent's implementation
        would 404 mid-run for something purely advisory."""
        provider = BedrockProvider.__new__(BedrockProvider)

        count = await provider.count_tokens([Message.user_text("hello there world")])

        assert count > 0


class TestMissingDependency:
    def test_it_names_the_extra_to_install(self, monkeypatch):
        """A bare ImportError for `boto3` does not tell anyone what to do."""
        import builtins

        real_import = builtins.__import__

        def no_bedrock(name, *args, **kwargs):
            if name == "anthropic" and args and "AsyncAnthropicBedrock" in (args[2] or ()):
                raise ImportError("no boto3")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_bedrock)

        with pytest.raises(RuntimeError, match=r"anthropic\[bedrock\]"):
            BedrockProvider("claude-opus-5", region="us-east-1")


class TestConstruction:
    def test_it_wires_the_bedrock_client_and_prefixed_id(self, monkeypatch):
        """No AWS call — just that the right client is built with the right
        region, and that capabilities resolve from the *bare* id."""
        captured = {}

        class FakeBedrockClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import anthropic

        monkeypatch.setattr(
            anthropic, "AsyncAnthropicBedrock", FakeBedrockClient, raising=False
        )

        provider = BedrockProvider("claude-opus-5", region="eu-west-1")

        assert provider.model == "anthropic.claude-opus-5"
        assert provider.region == "eu-west-1"
        assert captured == {"aws_region": "eu-west-1"}
        assert provider.capabilities.max_output_tokens > 0

    def test_explicit_credentials_are_passed_through(self, monkeypatch):
        captured = {}

        class FakeBedrockClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import anthropic

        monkeypatch.setattr(
            anthropic, "AsyncAnthropicBedrock", FakeBedrockClient, raising=False
        )

        BedrockProvider(
            "claude-opus-5",
            region="us-east-1",
            aws_access_key="AKIA",
            aws_secret_key="secret",
        )

        assert captured["aws_access_key"] == "AKIA"
        assert captured["aws_secret_key"] == "secret"

    def test_omitted_credentials_fall_through_to_the_aws_chain(self, monkeypatch):
        """A caller running inside AWS should not have to plumb credentials —
        passing empty strings would defeat boto3's own resolution."""
        captured = {}

        class FakeBedrockClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import anthropic

        monkeypatch.setattr(
            anthropic, "AsyncAnthropicBedrock", FakeBedrockClient, raising=False
        )

        BedrockProvider("claude-opus-5", region="us-east-1")

        assert "aws_access_key" not in captured
        assert "aws_secret_key" not in captured
