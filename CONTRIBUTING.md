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
- [How to add a built-in Task](#how-to-add-a-built-in-task)
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
pytest -q                  # 174 tests, should all pass
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
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_providers.py::test_anthropic_live -s

# OpenAI-compatible live smoke test (requires a running local server)
OPENAI_BASE_URL=http://localhost:1234/v1 pytest tests/test_providers.py::test_openai_live -s
```

The CI suite runs only the mocked tests (no live provider required).

---

## Project structure

```
neurosurfer/
  cli/          interactive REPL, slash commands, non-interactive subcommands
  llm/          canonical types + Provider protocol + Anthropic/OpenAI adapters
  core/         agent loop, permissions, context management, durable state, rails
  tools/        built-in tool library
  agents/       built-in specialist sub-agents
  tasks/        Task model, registry, runner, policy enforcement, built-in YAMLs
  prompts/      system prompt assembly
  observability/  run transcripts, structured logging
tests/          unit + integration + e2e tests
docs/           user-facing documentation
fixtures/       sample repo used in e2e tests
```

---

## How to add a new Tool

1. Create `neurosurfer/tools/<your_tool>.py`. Subclass `Tool` from
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

2. Register it in `neurosurfer/tools/__init__.py` (add to `ALL_TOOLS`).

3. Add tests in `tests/test_tools.py` covering the happy path and at least one
   error path (errors must surface as `ToolResult(is_error=True)`, not
   exceptions).

4. Document it in `docs/TASKS.md` under the guardrails/tools section if it
   introduces new guardrail surface.

---

## How to add a new Provider adapter

The engine depends only on the `Provider` protocol defined in
`neurosurfer/llm/base.py`. A new adapter must implement:

```python
class MyProvider:
    async def stream(self, messages, system, tools, config) -> AsyncIterator[CanonicalEvent]: ...
    async def complete(self, messages, system, tools, config) -> CanonicalResponse: ...
    async def count_tokens(self, messages, system, tools) -> int: ...
    @property
    def capabilities(self) -> ProviderCapabilities: ...
```

Steps:
1. Create `neurosurfer/llm/<provider>_provider.py`.
2. Map your provider's wire format to/from the canonical types in
   `neurosurfer/llm/types.py` (canonical content blocks: `text`, `thinking`,
   `tool_use`, `tool_result`; canonical events: `text_delta`, `tool_use`,
   `thinking_delta`, `usage`, `stop`).
3. Register it in `neurosurfer/llm/registry.py`.
4. Add a provider-parity test: run the same multi-turn, tool-calling scripted
   conversation through both your adapter and the existing Anthropic adapter
   (using the fake transport) and assert identical canonical events.
5. Update `docs/PROVIDERS.md` with setup instructions.

---

## How to add a built-in Task

Built-in Tasks are YAML files in `neurosurfer/tasks/builtin/`. They are
discovered automatically on startup and appear in `neurosurfer task list`.

1. Create `neurosurfer/tasks/builtin/<name>.yaml` following the Task YAML
   schema in [docs/TASKS.md](docs/TASKS.md).
2. Validate it passes the policy ceiling: `neurosurfer task show <name>`.
3. Add an e2e test in `tests/test_e2e.py` that runs the task against the
   `fixtures/sample_repo/` using a scripted provider.
4. Document the task in [docs/TASKS.md](docs/TASKS.md) and list it in the
   README's Built-in Tasks section.

---

## Pull request process

1. **Branch** from `main`: `git checkout -b feat/my-feature`.
2. **Keep PRs focused.** One logical change per PR; split unrelated fixes.
3. **Write tests** for any new behavior. The CI suite must remain green.
4. **Update docs** if you add or change user-visible behavior (README, relevant
   `docs/` file).
5. **Fill in the PR template** — summary, test plan, and checklist.
6. **CI must pass** — ruff, pytest, and the package build (twine check).
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
