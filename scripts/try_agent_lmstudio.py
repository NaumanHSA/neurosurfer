#!/usr/bin/env python
"""Minimal harness: drive the native AgenticLoop against a local LM Studio server.

Builds an OpenAI-compatible Provider pointed at LM Studio, gives the agent a
small read-only tool pool (list_dir / read_file / search / finish) and a tiny
task, then streams the run so you can watch it think, call tools, and finish.

Prereqs:
  1. LM Studio running with a tool-calling model loaded, local server started
     (default endpoint http://localhost:1234/v1). Set MODEL below to that model id.
  2. Run inside the project env:

        conda run -n LLMs python scripts/try_agent_lmstudio.py

     Override endpoint/model without editing the file:

        LMSTUDIO_BASE_URL=http://localhost:1234/v1 \
        LMSTUDIO_MODEL="qwen2.5-7b-instruct" \
        conda run -n LLMs python scripts/try_agent_lmstudio.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from neurosurfer.agents import Guardrails, events
from neurosurfer.agents.agentic_loop import AgenticLoop
from neurosurfer.llm.providers.openai import OpenAICompatProvider
from neurosurfer.tools.base import IOHandler, ToolPool, WriteChoice
from neurosurfer.tools.builtin import FinishTool, ListDirTool, ReadFileTool, SearchTool

# ── config ────────────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("LMSTUDIO_MODEL", "google/gemma-4-12b-qat")
CONTEXT_WINDOW = int(os.environ.get("LMSTUDIO_CONTEXT", "32768"))

TASK = (
    "List the files in the current directory, read pyproject.toml, and tell me "
    "in 2-3 sentences what this project is and what its CLI entry point is. "
    "Call finish when done."
)


# ── a headless, auto-approving IO handler (no human in the loop) ────────────────
class AutoIO(IOHandler):
    """Non-interactive IOHandler: prints notifications, auto-approves everything.

    Fine for a read-only demo. Do NOT reuse for tasks that write or run shell
    without understanding that this blindly says yes.
    """

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        print(f"\n[ask] {question}  -> {(options or ['']) [0]!r}")
        return (options or [""])[0]

    async def request_plan_approval(self, plan: str) -> tuple[bool, str]:
        return True, ""

    async def request_shell_approval(self, command: str, reason: str) -> bool:
        return True

    async def request_write_approval(self, path: str, summary: str) -> WriteChoice:
        return "once"

    def notify(self, message: str) -> None:
        print(f"[notify] {message}")


async def main() -> None:
    provider = OpenAICompatProvider(
        base_url=BASE_URL,
        api_key="lm-studio",  # LM Studio ignores the key
        model=MODEL,
        context_window=CONTEXT_WINDOW,
        max_output_tokens=8000,
    )

    tools = ToolPool([ReadFileTool(), ListDirTool(), SearchTool(), FinishTool()])

    agent = AgenticLoop(
        provider=provider,
        tools=tools,
        system_prompt=(
            "You are a concise coding assistant. Use the tools to inspect the "
            "repository, then answer. Keep tool calls minimal."
        ),
        guardrails=Guardrails(),
        io=AutoIO(),
        cwd=Path.cwd(),
    )

    print(f"→ provider: {BASE_URL}  model: {MODEL}")
    print(f"→ task: {TASK}\n" + "─" * 70)

    async for ev in agent.run(TASK):
        if isinstance(ev, events.TextDelta):
            print(ev.text, end="", flush=True)
        elif isinstance(ev, events.ThinkingDelta):
            print(ev.text, end="", flush=True)
        elif isinstance(ev, events.ToolStarted):
            print(f"\n  🛠  {ev.name}({ev.args})")
        elif isinstance(ev, events.ToolFinished):
            preview = (ev.result.content or "").strip().replace("\n", " ")[:120]
            print(f"  ✓ {ev.name} → {preview}")
        elif isinstance(ev, events.RunFinished):
            print("\n" + "─" * 70)
            print(f"✅ status={ev.status}")
            if ev.report:
                print(f"report: {ev.report}")
        elif isinstance(ev, events.AgentError):
            print(f"\n❌ error: {ev}")

    print(f"\n(turns={agent.turns}, tokens={agent.usage.total()})")


if __name__ == "__main__":
    asyncio.run(main())
