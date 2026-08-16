"""Claude on Amazon Bedrock.

**A subclass, not a second implementation.** Bedrock serves the same Messages
API through a different client and a different id scheme, so everything that
makes `AnthropicProvider` work — the message translation, the tool translation,
the streaming event mapping, the thinking handling — is already correct here.
Copying it would create two things to keep in step, and the second one would
drift; there is nothing in this file that is not about the client or the id.

Two differences, and they are the whole file:

* **Auth is AWS**, not an Anthropic key: SigV4 over boto3's credential chain
  (env vars, shared profile, instance role). There is no API key to pass.
* **Model ids carry an `anthropic.` prefix** — `anthropic.claude-opus-5`. A
  first-party id sent to Bedrock is a 400, so it is added when absent rather
  than left for the caller to remember.

Needs `anthropic[bedrock]`, which brings boto3. The import is deferred so that
importing this module — or anything that merely mentions the provider — does
not require it.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider, anthropic_capabilities

__all__ = ["BedrockProvider", "to_bedrock_model_id"]

#: The prefix Bedrock puts on every Anthropic-published model.
_PREFIX = "anthropic."


def to_bedrock_model_id(model: str) -> str:
    """A first-party model id → its Bedrock form, idempotently.

    Adding the prefix here rather than requiring it from the caller means the
    same configuration string works against either provider — which is the point
    of having one `Provider` interface at all.
    """
    name = model.strip()
    return name if name.startswith(_PREFIX) else f"{_PREFIX}{name}"


def from_bedrock_model_id(model: str) -> str:
    """The bare first-party id, for capability and price lookups."""
    return model[len(_PREFIX) :] if model.startswith(_PREFIX) else model


class BedrockProvider(AnthropicProvider):
    """Claude through Amazon Bedrock.

    ``region`` is required — unlike the first-party client there is no sensible
    default, and a wrong region fails at request time with an unhelpful error
    rather than at construction.
    """

    def __init__(
        self,
        model: str,
        *,
        region: str,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropicBedrock
        except ImportError as e:
            raise RuntimeError(
                "Bedrock support needs the AWS extra — pip install "
                "'anthropic[bedrock]' (it brings boto3)."
            ) from e

        kwargs: dict[str, object] = {"aws_region": region}
        # Left unset, the client falls back to boto3's credential chain —
        # environment, shared profile, instance role — which is what a caller
        # running inside AWS expects and should not have to plumb through.
        if aws_access_key:
            kwargs["aws_access_key"] = aws_access_key
        if aws_secret_key:
            kwargs["aws_secret_key"] = aws_secret_key
        if aws_session_token:
            kwargs["aws_session_token"] = aws_session_token
        if base_url:
            kwargs["base_url"] = base_url

        self._client = AsyncAnthropicBedrock(**kwargs)  # type: ignore[arg-type]
        self.model = to_bedrock_model_id(model)
        self.region = region
        # Capabilities are a property of the model, not of who serves it, so
        # they are resolved from the bare id — `anthropic.claude-opus-5` would
        # match nothing in a table keyed on first-party names.
        self.capabilities = anthropic_capabilities(from_bedrock_model_id(self.model))

    async def count_tokens(self, messages, system=None, tools=None) -> int:
        """Bedrock has no token-counting endpoint — estimate locally.

        The parent's implementation calls `client.messages.count_tokens`, which
        does not exist here; a 404 mid-run for something advisory is a bad trade.
        """
        from ..tokens import estimate_messages_tokens

        return estimate_messages_tokens(messages, system, tools or [])
