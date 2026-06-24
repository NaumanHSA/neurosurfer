"""ArchitectBuilder — loads the built-in architect WorkflowPackage and runs it.

Usage::

    from neurosurfer.graph.builder.build import ArchitectBuilder

    builder = ArchitectBuilder(provider)
    result = await builder.run("Build a workflow that summarises GitHub PRs daily")
    print(result)  # path to the registered workflow
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neurosurfer.graph.workflow.package import WorkflowPackage, load_package
from neurosurfer.graph.workflow.runner import WorkflowRunner
from neurosurfer.llm.base import Provider

_PACKAGE_DIR = Path(__file__).parent / "package"


class ArchitectBuilder:
    """Drives the built-in Architect workflow to design and register a new workflow.

    Parameters
    ----------
    provider:
        The LLM provider powering all agent nodes in the architect graph.
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._pkg: WorkflowPackage | None = None

    @property
    def package(self) -> WorkflowPackage:
        if self._pkg is None:
            self._pkg = load_package(_PACKAGE_DIR)
        return self._pkg

    async def run(
        self,
        user_intent: str,
        *,
        answers: dict[str, Any] | None = None,
        progress_callback: Any = None,
        on_node_event: Any = None,
        approve_tool: Any = None,
        notify: Any = None,
    ) -> str:
        """Run the architect on *user_intent* and return the registered package path.

        Parameters
        ----------
        user_intent:
            Plain-English description of what the desired workflow should do.
        answers:
            Pre-collected clarifying answers from the conversational front-end
            (question_id → answer text).  When provided, the ``clarify`` function
            node skips its interactive ``input()`` prompts.
        progress_callback:
            Optional ``(node_id, status, duration_ms)`` post-run summary callback.
        on_node_event:
            Optional ``(node_id, status)`` callback fired live as nodes run.
        approve_tool:
            Optional async ``(ToolDraft, SandboxResult) -> bool`` approval callback.
            Required only if the designed workflow turns out to need a tool that does
            not exist yet (a capability gap); the Architect will author it, validate
            it in a sandbox, and register it only if this callback approves.

        Returns
        -------
        str
            Absolute path to the registered workflow directory.
        """
        from neurosurfer.tools.registry import format_workflow_tool_catalog

        runner = WorkflowRunner(
            self._provider,
            allowed_tools={"web_search", "write_workflow_node"},
        )
        inputs: dict[str, Any] = {
            "user_intent": user_intent,
            # The real tool catalog generated workflows may use. Interpolated into
            # the plan / write_nodes prompts via `{available_tools}` so the Architect
            # never invents tool names.
            "available_tools": format_workflow_tool_catalog(),
        }
        if answers:
            inputs["answers"] = answers

        # Always capture a debug trace of the build so we can inspect exactly what
        # each LLM node (discover / plan / write_nodes) was prompted with and returned.
        import time

        from neurosurfer.config.paths import traces_dir

        trace_path = traces_dir() / f"architect-build-{int(time.time())}.json"
        say = notify or (lambda _m: None)

        result = runner.run(
            self.package,
            inputs=inputs,
            progress=progress_callback,
            on_node_event=on_node_event,
            trace_path=trace_path,
        )
        if trace_path.exists():
            say(f"Build trace saved to {trace_path}")

        # Hard failure inside assemble (or upstream) — surface it.
        assemble_res = result.nodes.get("assemble")
        if assemble_res is not None and assemble_res.error:
            raise RuntimeError(f"Workflow build failed: {assemble_res.error}")

        out = result.final.get("assemble", "")
        out = str(out) if out else ""

        # Staged-but-not-registered: assemble withheld registration because the package
        # didn't pass validation. Re-validate here and either render a clean error
        # (hard errors) or author the missing tool(s) with approval, then register.
        from neurosurfer.graph.workflow.validate import DEFER_MARKER

        if out.startswith(DEFER_MARKER):
            project_dir = out[len(DEFER_MARKER):]
            return await self._finalize_staged(project_dir, approve_tool, notify)

        return out or "(assemble node produced no output)"

    async def _finalize_staged(
        self, project_dir: str, approve_tool: Any, notify: Any = None
    ) -> str:
        """Finalize a staged package: render hard errors, or author gaps, then register."""
        from neurosurfer.graph.workflow.registry import WorkflowRegistry
        from neurosurfer.graph.workflow.validate import validate_package

        from .tool_author import ToolAuthor

        say = notify or (lambda _m: None)
        pkg = load_package(Path(project_dir))
        report = validate_package(pkg)

        if report.errors:
            # Hard errors (e.g. an unauthorable function node, a typo'd tool) can't be
            # auto-fixed during build. Surface them cleanly — no traceback.
            raise RuntimeError(
                "The designed workflow has problems that can't be auto-fixed:\n"
                + report.summary()
            )
        if not report.gaps:
            return str(WorkflowRegistry().save(pkg))  # passed on re-check — register
        if approve_tool is None:
            raise RuntimeError(
                "Workflow needs tool(s) that do not exist and no approval handler "
                "was provided:\n" + report.summary()
            )

        gap_names = sorted({g.subject for g in report.gaps if g.subject})
        say(
            f"The design needs {len(gap_names)} tool(s) that don't exist yet: "
            f"{', '.join(gap_names)}. Authoring them…"
        )

        author = ToolAuthor(self._provider)
        # De-dup gaps by tool name (a tool may be referenced by several nodes).
        seen: set[str] = set()
        for gap in report.gaps:
            name = gap.subject or ""
            if not name or name in seen:
                continue
            seen.add(name)
            spec = self._gap_to_spec(gap, pkg)
            meta = await author.author(spec, approve=approve_tool, notify=say)
            if meta is None:
                if getattr(author, "rejected", False):
                    raise RuntimeError(
                        f"Tool '{name}' was declined. Workflow not registered. "
                        f"Staged for inspection at: {project_dir}"
                    )
                raise RuntimeError(
                    f"Couldn't auto-write a working '{name}' tool "
                    f"({getattr(author, 'last_failure', 'unknown error')}). "
                    f"Workflow not registered. Staged at: {project_dir}\n"
                    f"Tip: rephrase so the node composes existing tools "
                    f"(read_file/list_dir/run_command/write_file)."
                )

        # Re-validate now that the new tools are discoverable.
        pkg = load_package(Path(project_dir))
        report2 = validate_package(pkg)
        if not report2.ok:
            raise RuntimeError(
                "Workflow still invalid after authoring tools:\n" + report2.summary()
            )
        dest = WorkflowRegistry().save(pkg)
        return str(dest)

    def _gap_to_spec(self, gap: Any, pkg: WorkflowPackage) -> Any:
        from .tool_author import ToolGapSpec

        node = next((n for n in pkg.graph.nodes if n.id == gap.node_id), None)
        purpose = ((node.purpose or node.goal or "") if node else "").strip()
        return ToolGapSpec(
            name=gap.subject or "",
            purpose=(
                f"Provide the '{gap.subject}' capability for workflow node "
                f"'{gap.node_id}'. Node purpose: {purpose}"
            ),
            context=purpose,
            source_workflow=pkg.name,
        )
