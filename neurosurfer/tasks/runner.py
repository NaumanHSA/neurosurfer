"""Task runner — boots an Agent from a TaskDefinition.

The runner:
1. Resolves the LLM provider (Task model override or global config).
2. Filters the full tool pool to the Task's ``tools`` allow-list.
3. Wires the ``SubAgentRunner`` spawn function into the ToolContext.
4. Opens a transcript for the run.
5. Accepts pre-collected inputs (CLI gathers them interactively before calling).
6. Returns an async generator of engine events (same shape as Agent.run).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Config
from ..agents import events
from ..agents.agentic_loop import AgenticLoop
from ..agents.context.manager import ContextManager
from ..agents.context.durable_state import DurableState
from ..agents.runtime.permissions import Guardrails
from ..agents.subagents.runner import SubAgentRunner
from ..llm.registry import build_provider
from ..observability.logging import get_logger
from ..observability.transcript import EventTranscript
from .definition import TaskDefinition
from .ingest import ingest_repo

if TYPE_CHECKING:
    from ..llm.base import Provider
    from ..tools.base import IOHandler, ToolPool

log = get_logger("tasks.runner")


def build_full_pool(
    cwd: Path, tasks_dir: Path | None = None
) -> ToolPool:
    """Assemble the full curated tool pool for the current working directory.

    ``tasks_dir`` is where ``register_task`` persists new Task YAMLs.
    When omitted it defaults under ``~/.neurosurfer/``.
    """
    from ..tools.base import ToolPool
    from ..tools.builtin import (
        ApplyEditTool,
        AskUserTool,
        BrowseTool,
        DataTool,
        FinishTool,
        HttpTool,
        ListDirTool,
        MemoryTool,
        ReadFileTool,
        RunCommandTool,
        SearchTool,
        SpawnAgentTool,
        TodoTool,
        WebSearchTool,
        WriteFileTool,
    )
    from ..app.tools import PresentPlanTool, RegisterTaskTool

    _tasks_dir = tasks_dir or (Path.home() / ".neurosurfer" / "tasks")
    return ToolPool([
        ReadFileTool(),
        ListDirTool(),
        SearchTool(),
        DataTool(),
        RunCommandTool(),
        WebSearchTool(),
        HttpTool(),
        BrowseTool(),
        WriteFileTool(),
        ApplyEditTool(),
        AskUserTool(),
        PresentPlanTool(),
        TodoTool(),
        SpawnAgentTool(),
        MemoryTool(),
        FinishTool(),
        RegisterTaskTool(_tasks_dir),
    ])


class TaskRunner:
    """Run a TaskDefinition as a full agent session.

    A ``provider`` may be injected (tests, or a pre-built provider); otherwise one
    is built from config + the Task's optional model override. If the Task declares
    a ``path_or_url`` input, its value is ingested (local path or git clone) and
    becomes the agent's working directory for the run.
    """

    def __init__(
        self, cfg: Config, cwd: Path | None = None, provider: Provider | None = None
    ) -> None:
        self._cfg = cfg
        self._cwd = cwd or Path.cwd()
        self._provider = provider

    def run(
        self,
        task: TaskDefinition,
        inputs: dict[str, str],
        io: IOHandler,
        *,
        transcript: EventTranscript | None = None,
        run_id: str | None = None,
        durable: DurableState | None = None,
        resume: bool = False,
    ) -> AsyncIterator[events.Event]:
        return self._run(
            task, inputs, io,
            transcript=transcript, run_id=run_id, durable=durable, resume=resume,
        )

    def _resolve_cwd(self, task: TaskDefinition, inputs: dict[str, str]):
        """If the Task has a path_or_url input, ingest it as the working dir.

        Returns (cwd, ingested) where ``ingested`` (if any) must be cleaned up.
        """
        for spec in task.inputs:
            if spec.type == "path_or_url":
                value = inputs.get(spec.name)
                if value:
                    ingested = ingest_repo(value)
                    return ingested.path, ingested
        return self._cwd, None

    async def _run(
        self,
        task: TaskDefinition,
        inputs: dict[str, str],
        io: IOHandler,
        *,
        transcript: EventTranscript | None = None,
        run_id: str | None = None,
        durable: DurableState | None = None,
        resume: bool = False,
    ) -> AsyncIterator[events.Event]:
        # 0. Resolve working directory (ingest a path_or_url input if present).
        cwd, ingested = self._resolve_cwd(task, inputs)
        if ingested is not None:
            log.info("ingested repository → %s", cwd)

        # Where durable state is persisted so an interrupted run is resumable.
        state_path = (
            self._cfg.observability.runs_state_dir() / f"{run_id}.json" if run_id else None
        )

        try:
            # 1. Provider (injected or built from config + Task model override)
            provider = self._provider or build_provider(
                self._cfg, model_override=task.model
            )

            # 2. Tool pool — filtered to the Task's allow-list
            full_pool = build_full_pool(cwd, tasks_dir=self._cfg.tasks.dir)
            tool_pool = full_pool.select(task.tools) if task.tools else full_pool

            # 3. Guardrails from the Task definition; honour output_dir input.
            guardrails: Guardrails = _resolve_guardrails(task, inputs)

            # 3b. Local-first reliability profile: clamp sub-agent fan-out for weak
            #     models and decide whether to inject the reasoning scaffold.
            from ..llm.capabilities import reliability_profile
            from ..prompts.base_agent import think_scaffold_section

            profile = reliability_profile(provider.capabilities)
            if profile.max_concurrent_subagents < guardrails.max_concurrent_subagents:
                guardrails = guardrails.model_copy(
                    update={"max_concurrent_subagents": profile.max_concurrent_subagents}
                )
            extra_sections: list[str] = []
            if profile.think_scaffold:
                extra_sections.append(think_scaffold_section())

            # 3c. Long-term memory (Pillar 1): build the store, retrieve the entries
            #     relevant to this run (global ∪ this agent), and inject them once via
            #     extra_sections — after the cacheable prefix, so the warm prompt cache
            #     is never busted. Any failure degrades silently; memory never breaks a run.
            memory_store = self._build_memory_store(task, inputs, extra_sections)
            if memory_store is not None and "memory" in (task.tools or []):
                from ..prompts.base_agent import memory_usage_section
                extra_sections.append(memory_usage_section())

            # 4. Sub-agent runner (spawn function wired into the ToolContext)
            sub_runner = SubAgentRunner(
                full_pool, provider, io=io, cwd=cwd, guardrails=guardrails
            )
            spawn_fn = sub_runner.make_spawn_fn(parent_depth=0)

            # 5. Durable state + context manager. On resume, a pre-loaded
            #    DurableState (plan/todos/decisions) is injected so the agent
            #    continues with its prior context intact.
            run_durable = durable if durable is not None else DurableState()
            context_manager = ContextManager(provider, durable=run_durable)

            # 6. System prompt — base sections (identity/tone/tools/planning/guardrails/env)
            #    wrapped around the Task-specific instructions.
            system_prompt = _build_system_prompt(
                task, inputs, guardrails=guardrails, cwd=cwd, model=provider.model,
                extra_sections=extra_sections,
            )

            # 7. Instantiate the agent
            from ..agents.runtime.permissions import PermissionMode

            # On resume with an already-approved plan, skip the plan gate so the
            # agent continues executing rather than re-presenting the plan.
            plan_already_approved = resume and bool(run_durable.plan_text)
            mode: PermissionMode = (
                "plan" if task.plan_required and not plan_already_approved else "default"
            )
            agent = AgenticLoop(
                provider=provider,
                tools=tool_pool,
                system_prompt=system_prompt,
                guardrails=guardrails,
                io=io,
                cwd=cwd,
                mode=mode,
                durable=run_durable,
                spawn=spawn_fn,
                context_manager=context_manager,
                depth=0,
                persist_scope=make_scope_persister(self._cfg, task.name),
                memory=memory_store,
                memory_agent=task.name,
                session_id=run_id,
                # The CLI renders the event stream itself (rich spinners + trace via
                # stream_events); the logging side-channel would duplicate it.
                verbose=False,
            )

            # 8. Initial user message: brief summary of activated inputs (or a
            #    resume directive when continuing an interrupted run).
            initial_msg = (
                _build_resume_message(task)
                if resume
                else _build_initial_message(task, inputs)
            )

            if transcript:
                transcript.record(
                    "run_resume" if resume else "run_start",
                    task=task.name, inputs=inputs, cwd=str(cwd),
                )

            async for ev in agent.run(initial_msg):
                if transcript:
                    transcript.record(ev.__class__.__name__, **_event_payload(ev))
                # Persist durable state each completed turn so an interrupt
                # (Ctrl-C) at any point leaves a resumable snapshot on disk.
                if state_path is not None and isinstance(ev, events.TurnCompleted):
                    run_durable.save(state_path)
                yield ev

            if state_path is not None:
                run_durable.save(state_path)
            # Capture: distill this run's durable decisions into candidate memories.
            self._distill_memory(memory_store, run_durable, agent=task.name, session_id=run_id)
            if transcript:
                transcript.record("run_end")
        finally:
            if ingested is not None:
                ingested.cleanup()

    # ── memory (Pillar 1) ─────────────────────────────────────────────────────
    def _build_memory_store(self, task, inputs, extra_sections):
        """Build the MemoryStore and append the retrieved `# Relevant memory` block.

        Returns the store (passed to the Agent for the in-run ``memory`` tool) or
        ``None`` when memory is disabled or retrieval fails — never raising.
        """
        if not self._cfg.memory.enabled:
            return None
        try:
            from ..memory.embeddings import get_embedder
            from ..memory.retrieval import retrieve
            from ..memory.store import MemoryStore

            store = MemoryStore(self._cfg.memory.dir)
            query = " ".join(
                [task.name, task.description, *[str(v) for v in inputs.values()]]
            )
            result = retrieve(
                store.all_in_scope(task.name),
                query,
                budget_tokens=self._cfg.memory.token_budget,
                embedder=get_embedder(self._cfg.memory.embeddings_backend),
            )
            if result.block:
                extra_sections.append(result.block)
                store.record_use(result.entry_ids)
            return store
        except Exception as e:  # noqa: BLE001 - memory must never break a run
            log.warning("memory retrieval skipped: %s", e)
            return None

    def _distill_memory(self, store, durable, *, agent: str, session_id: str | None) -> None:
        if store is None or not self._cfg.memory.enabled:
            return
        try:
            from ..memory.distill import distill_run

            distill_run(store, durable, agent=agent, session_id=session_id)
        except Exception as e:  # noqa: BLE001 - capture is best-effort
            log.warning("memory distill skipped: %s", e)


def make_scope_persister(cfg: Config, task_name: str):
    """Return a callback that persists a newly-approved write folder onto a Task.

    When the user answers "always" to an out-of-scope write prompt, the engine
    calls this to add the folder to the Task's ``write_scope`` and re-save it, so
    the approval survives across sessions. Failures (e.g. a built-in Task or a
    policy-rejected root) are logged, not raised — the session-wide widening in
    ``permissions`` still applies.
    """
    from .registry import TaskRegistry

    def _persist(entry: str) -> None:
        try:
            reg = TaskRegistry(cfg.tasks.dir)
            td = reg.get(task_name)
            if td.is_protected:
                log.debug("scope persistence skipped: '%s' is a %s task", task_name, td.kind)
                return
            if entry not in td.guardrails.write_scope:
                td.guardrails.write_scope.append(entry)
                reg.save(td)  # validates against the PolicyCeiling
                log.info("persisted write scope '%s' onto task '%s'", entry, task_name)
        except Exception as e:  # noqa: BLE001 - persistence is best-effort
            log.warning("could not persist write scope '%s' for '%s': %s", entry, task_name, e)

    return _persist


def _resolve_guardrails(task: TaskDefinition, inputs: dict[str, str]) -> Guardrails:
    """Return guardrails, overriding write_scope if the task has an output_dir input."""
    guardrails = task.guardrails
    for spec in task.inputs:
        if spec.name == "output_dir" and spec.type == "path":
            val = inputs.get("output_dir") or spec.default or "docs/"
            scope = val.rstrip("/") + "/"
            guardrails = guardrails.model_copy(update={"write_scope": [scope]})
            break
    return guardrails


def _build_system_prompt(
    task: TaskDefinition,
    inputs: dict[str, str],
    *,
    guardrails: Guardrails,
    cwd: Path,
    model: str,
    extra_sections: list[str] | None = None,
) -> str:
    """Layer base sections (identity/tone/tools/planning/guardrails/env) around the task body.

    The base sections are static and cacheable on Anthropic; task-specific
    instructions go under "# Your task". ``extra_sections`` (reliability scaffold
    now, retrieved memory in a later pillar) are appended *after* the cacheable
    prefix so they never invalidate the warm cache.
    """
    from ..prompts.system import build_system_prompt as _assemble

    # Substitute <output_dir> placeholder with the resolved value.
    output_dir = inputs.get("output_dir", "docs/").rstrip("/") + "/"
    task_instructions = task.system_prompt.replace("<output_dir>", output_dir)
    if inputs:
        task_instructions += "\n\n## Provided inputs\n"
        task_instructions += "\n".join(f"- **{k}**: {v}" for k, v in inputs.items())

    return _assemble(
        task_instructions=task_instructions,
        guardrails=guardrails,
        cwd=cwd,
        model=model,
        extra_sections=extra_sections,
    )


def _build_initial_message(task: TaskDefinition, inputs: dict[str, str]) -> str:
    if inputs:
        kvs = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
        return f"Run the task '{task.name}' with inputs: {kvs}."
    return f"Run the task '{task.name}'."


def _build_resume_message(task: TaskDefinition) -> str:
    return (
        f"Resume the previously interrupted run of task '{task.name}'. Your durable "
        "state — the approved plan, todos, and recorded decisions — has been restored "
        "and appears in the system context. Review what is already done (completed "
        "todos), do not redo finished work, and continue from where you left off."
    )


def _event_payload(ev: events.Event) -> dict:
    result: dict = {}
    for attr in ("text", "status", "report", "id", "name", "input", "result"):
        val = getattr(ev, attr, None)
        if val is not None:
            result[attr] = str(val)[:500]
    return result
