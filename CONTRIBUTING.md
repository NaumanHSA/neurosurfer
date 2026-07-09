# Contributing to neurosurfer

Thank you for taking the time to contribute! This document covers everything
you need to get a development environment running, our code conventions, and
the process for submitting changes.

---

## Table of Contents

- [Development setup](#development-setup)
- [Code style](#code-style)
- [Running tests](#running-tests)
- [Project structure](#project-structure)
- [How to add a new Tool](#how-to-add-a-new-tool)
- [How to add a new Provider adapter](#how-to-add-a-new-provider-adapter)
- [Pull request process](#pull-request-process)
- [Reporting bugs](#reporting-bugs)

---

## Development setup

**Requirements:** Python 3.11+, git.

```bash
git clone https://github.com/NaumanHSA/neurosurfer
cd neurosurfer

# Create and activate a virtual environment (conda or venv)
python -m venv .venv && source .venv/bin/activate
# or: conda create -n neurosurfer python=3.12 && conda activate neurosurfer

# Install the package in editable mode with all dev dependencies
pip install -e ".[dev]"

# Copy and configure the environment file
cp .env.example .env
# Set at least one provider: ANTHROPIC_API_KEY or OPENAI_BASE_URL + MODEL
```

Verify everything works:

```bash
neurosurfer doctor        # checks provider reachability
pytest -q                  # should all pass
```

---

## Code style

We use **ruff** for linting/formatting and **mypy** for type checking.

```bash
ruff check neurosurfer tests   # lint
ruff format neurosurfer tests  # format
mypy neurosurfer               # type check
```

All three must pass cleanly before a PR is merged. CI enforces this on every
pull request.

**Key conventions:**
- Type-annotate all public functions and method signatures.
- No bare `except:` — catch specific exception types.
- Tool errors are returned as `ToolResult(is_error=True, ...)`, never raised.
- Provider adapters depend only on `neurosurfer.llm` canonical types — never
  import `anthropic` or `openai` outside their respective adapter files.
- Keep comments minimal; write them only when the *why* is non-obvious.

---

## Running tests

```bash
pytest -q                          # full suite
pytest tests/test_tools.py -q      # a specific file
pytest -k "test_compaction" -q     # a specific test
pytest -q --tb=short               # shorter tracebacks
```

**Live provider tests** are gated by environment variables and skipped when the
provider isn't configured:

```bash
# Anthropic live smoke test
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_provider_parity.py -k live -s

# OpenAI-compatible live smoke test (requires a running local server)
OPENAI_BASE_URL=http://localhost:1234/v1 pytest tests/test_provider_parity.py -k live -s
```

The CI suite runs only the mocked tests (no live provider required).

---

## Project structure

```
neurosurfer/
  llm/            canonical types + Provider protocol + Anthropic/OpenAI adapters
  agents/         AgenticLoop, ReactAgent, Agent (one-shot), sub-agent runner,
                  context management, permissions/guardrails
  tools/          built-in tool library + tool-pool registry
  rag/            ingest, chunk, embed, retrieve, context injection
  vectorstores/   vector store backends (in-memory, Chroma)
  embeddings/     embedding provider adapters
  graph/          DAG engine (Graph, GraphExecutor) + persisted Workflow packages
  architect/      natural-language-to-Workflow builder (schemas, nodes, tool-author)
  mcp/            Model Context Protocol client (session, manager, tool bridging)
  app/            interactive CLI, built-in sub-agents/tools, OpenAI-compatible
                  gateway server
  config/         .env/profile loading, provider/MCP/observability config
  cache/          provider- and embedder-response caching (memory/disk backends)
  observability/  run transcripts, structured logging, trace context
  tracing/        span/tracer primitives + pluggable trace exporters
  prompts/        system prompt assembly
tests/            unit + integration tests
docs/             user-facing documentation (mkdocs)
tutorials/        Colab-ready notebooks
```

---

## How to add a new Tool

1. Create `neurosurfer/tools/builtin/<your_tool>.py`. Subclass `Tool` from
   `neurosurfer.tools.base`:

   ```python
   from pydantic import BaseModel
   from neurosurfer.tools.base import Tool, ToolResult, ToolContext

   class MyToolInput(BaseModel):
       path: str
       flag: bool = False

   class MyTool(Tool):
       name = "my_tool"
       description = "One sentence describing what this tool does."
       input_schema = MyToolInput
       is_read_only = True
       is_concurrency_safe = True
       is_destructive = False

       async def call(self, args: MyToolInput, ctx: ToolContext) -> ToolResult:
           try:
               result = do_something(args.path)
               return ToolResult(content=result)
           except Exception as e:
               return ToolResult(content=str(e), is_error=True)
   ```

2. Register it in `neurosurfer/tools/registry.py` (`all_tools()`), so it's
   picked up by `default_pool()` / `build_pool()`.

3. Add tests in `tests/test_tools.py` (or a subsystem-specific file, e.g.
   `tests/test_general_tools.py`) covering the happy path and at least one
   error path (errors must surface as `ToolResult(is_error=True)`, not
   exceptions).

4. Document it in [docs/guides/tools.md](docs/guides/tools.md).

---

## How to add a new Provider adapter

The engine depends only on the `Provider` protocol defined in
`neurosurfer/llm/base.py`. A new adapter must implement:

```python
class MyProvider(Provider):
    def stream(self, messages, system, tools, config) -> AsyncIterator[StreamEvent]: ...
    async def count_tokens(self, messages, system=None, tools=None) -> int: ...
    capabilities: ProviderCapabilities
    model: str
```

`complete()` is provided by the base class (it drains `stream()` and returns
the final `CanonicalResponse` carried by the terminal `Done` event) — you
don't need to override it.

Steps:
1. Create `neurosurfer/llm/providers/<provider>.py`.
2. Map your provider's wire format to/from the canonical types in
   `neurosurfer/llm/types.py` (canonical content blocks: `text`, `thinking`,
   `tool_use`, `tool_result`; canonical stream events: `TextDelta`,
   `ThinkingDelta`, `ToolUseStart`, `ToolUseArgsDelta`, `Done`).
3. Wire it into `build_provider()` / `build_provider_from_profile()` in
   `neurosurfer/llm/registry.py`.
4. Add a provider-parity test in `tests/test_provider_parity.py`: run the same
   multi-turn, tool-calling scripted conversation through both your adapter
   and the existing Anthropic adapter (using a fake transport) and assert
   equivalent canonical events.
5. Update [docs/guides/providers.md](docs/guides/providers.md) with setup
   instructions.

---

## Pull request process

1. **Branch** from `main`: `git checkout -b feat/my-feature`.
2. **Keep PRs focused.** One logical change per PR; split unrelated fixes.
3. **Write tests** for any new behavior. The CI suite must remain green.
4. **Update docs** if you add or change user-visible behavior (README, relevant
   `docs/` file).
5. **Fill in the PR template** — summary, test plan, and checklist.
6. **CI must pass** — ruff, mypy, pytest, and the package build (twine check).
7. A maintainer will review and merge. Squash merges are used to keep `main`
   history clean.

---

## Reporting bugs

Use [GitHub Issues](https://github.com/NaumanHSA/neurosurfer/issues) and
select the **Bug Report** template. Include:

- Your OS and Python version.
- The provider and model you're using.
- The full error output (redact your API key).
- Steps to reproduce.

For security vulnerabilities, see [SECURITY.md](SECURITY.md).
