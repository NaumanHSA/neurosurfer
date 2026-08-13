# Providers

A **provider** is Neurosurfer's adapter to an LLM. Every provider implements the same
`Provider` protocol (`neurosurfer.llm`), so agents, tools, and the gateway work unchanged whether
you're calling Anthropic, OpenAI, or a local OpenAI-compatible server.

```python
from neurosurfer.llm import Provider  # the protocol every provider satisfies
```

## Anthropic

```python
import os
from neurosurfer.llm.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-opus-4-8",
)
```

## OpenAI

```python
import os
from neurosurfer.llm.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o",
)
```

## Any OpenAI-compatible server

Ollama, LM Studio, vLLM, and llama.cpp all expose an OpenAI-compatible API. Point
`OpenAICompatProvider` at the server's `base_url`. Local models don't advertise their context size,
so pass `context_window` explicitly:

```python
from neurosurfer.llm.providers.openai import OpenAICompatProvider

provider = OpenAICompatProvider(
    base_url="http://localhost:11434/v1",   # e.g. Ollama
    api_key="not-needed",                    # most local servers ignore the key
    model="qwen2.5:7b",
    context_window=32_768,                   # match your model's real context size
)
```

Common `context_window` values: `4_096`, `8_192`, `16_384`, `32_768`, `65_536`, `131_072`.

!!! note "Only two adapters are first-class"
    Neurosurfer ships exactly two provider adapters — **Anthropic** and **OpenAI-compatible**. vLLM,
    Ollama, LM Studio, and llama.cpp are **not** separate providers; they're reached through
    `OpenAICompatProvider` by setting `base_url` (env: `OPENAI_BASE_URL`). If a server speaks the
    OpenAI API, it works here.

!!! tip "Native tool-calling vs. ReAct"
    `AgenticLoop` uses the provider's **native** function-calling API. If your local model doesn't
    support tool calls, use [`ReactAgent`](agents.md) instead, which drives tools by parsing text.

## Building a provider from config

`build_provider` constructs the active provider from a `Config` (which reads `.env` / environment
variables such as `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`):

```python
from neurosurfer.config import Config
from neurosurfer.llm import build_provider

provider = build_provider(Config())
```

This is the same mechanism the [CLI](../cli/index.md) uses for its provider profiles.

## Capabilities

Providers expose a capability descriptor so agents can adapt (e.g. whether the model supports native
tools or vision):

```python
from neurosurfer.llm import anthropic_capabilities, openai_capabilities

caps = anthropic_capabilities("claude-opus-4-8")
```

## Google Gemini

```python
from neurosurfer.llm.providers.gemini import GeminiProvider

provider = GeminiProvider(model="gemini-2.5-flash")   # GEMINI_API_KEY or GOOGLE_API_KEY
```

Speaks Gemini's native REST API over `httpx` — no extra dependency. Native tool calling, thinking
(surfaced as `ThinkingDelta`, separate from the answer), vision, and a real `countTokens` endpoint.

Gemini's wire format differs from the other two in four places the adapter handles for you: the
assistant role is called `model`, the system prompt is out-of-band, tool arguments arrive already
parsed, and tool results are matched to their call by **function name** rather than call id.
Thinking is not replayed on later turns — Gemini will not accept it back.

## Claude on Amazon Bedrock

```python
from neurosurfer.llm.providers.bedrock import BedrockProvider

provider = BedrockProvider("claude-opus-5", region="us-east-1")
```

Needs the `bedrock` extra (`pip install "neurosurfer[bedrock]"`, which brings boto3). Credentials
come from boto3's usual chain — environment, shared profile, instance role — unless you pass them
explicitly.

The `anthropic.` model-id prefix Bedrock requires is added for you, so the same model string works
against either provider. Bedrock has no token-counting endpoint, so `count_tokens` estimates locally.

## What a run cost

Token counts are priced per model, so a run reports money rather than only tokens:

```python
result = await agent.run_collect("...")
result.cost()                    # 0.0043, or None if the model is unpriced

graph_result.total_cost()        # each node priced at its own model
graph_result.execution_summary() # "… 7 nodes — 7 ok, 0 failed, 0 skipped — $0.04"
```

Rates live in `neurosurfer.llm.pricing.PRICES` — a plain dict you can extend or replace if you have
negotiated pricing:

```python
from neurosurfer.llm.pricing import PRICES, ModelPrice
PRICES["my-model"] = ModelPrice(input=0.5, output=1.5)   # $ per million tokens
```

An **unpriced model costs `None`, not `$0.00`** — reporting zero for a model missing from the table
would quietly under-report a bill. Sub-cent runs keep their digits (`$0.0043`) for the same reason.

## Canonical types & streaming

All providers speak the same canonical types (`Message`, `CanonicalResponse`, `StreamEvent`,
`TextDelta`, `ThinkingDelta`, `ToolUseBlock`, `Usage`, …) from `neurosurfer.llm`. Retry helpers
(`with_retry`, `is_retryable_error`) and token math (`estimate_messages_tokens`, `effective_window`,
`auto_compact_threshold`) live in the same package. You rarely call these directly — agents do — but
they're available when you need low-level control.
