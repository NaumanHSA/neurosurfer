"""Tool-author engine: generate, sandbox-test, and (on approval) register a new tool.

Phase E4 (authoring) + E5 (sandbox + test + approval gate).

When the validation gate (E1) reports a *capability gap* — a workflow node needs a
tool no registered tool provides — this engine asks the LLM to author a real
``neurosurfer.tools.base.Tool`` subclass, then **proves it is safe and conformant
before it is ever registered**:

    generate → static scan → sandbox check (subprocess) → USER APPROVAL → persist

Security contract (non-negotiable):
- The candidate's module body is executed only inside a resource-limited child
  process (`sys.executable`, wall-clock timeout), never in-process.
- Nothing is written to the live ``~/.neurosurfer/tools/`` directory until the
  user explicitly approves, having seen the code and the sandbox result. There is
  no code path that auto-registers generated code.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import GenerationConfig, Message
from neurosurfer.tools.generated import (
    GeneratedToolMeta,
    GeneratedToolsConfig,
    save_generated_tool,
)

logger = logging.getLogger(__name__)

# Tokens that warrant a flag in the approval prompt. Not a hard block — the human
# decides — but they must be surfaced, not buried.
_RISKY_PATTERNS = (
    "os.system",
    "subprocess",
    "eval(",
    "exec(",
    "__import__",
    "shutil.rmtree",
    "socket.",
    "pickle.loads",
    "rm -rf",
)

_SANDBOX_TIMEOUT_S = 15


@dataclass
class ToolGapSpec:
    """What a missing capability needs to do, derived from a failing node + gap.

    When sourced from the Architect's CapabilityPlan it also carries a functional
    *test plan* (``test_setup`` / ``test_args`` / ``expected_behavior``) so the
    sandbox can actually run the tool, not just import it.
    """

    name: str
    purpose: str
    inputs: list[str] = field(default_factory=list)
    context: str = ""
    source_workflow: str | None = None
    test_setup: str = ""
    test_args: dict = field(default_factory=dict)
    expected_behavior: str = ""


@dataclass
class SandboxResult:
    """Outcome of validating a candidate tool (static scan + sandbox import/test)."""

    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    functional_summary: str = ""  # result of actually calling the tool, if tested

    def render(self) -> str:
        lines = [f"  {'✓' if v else '✗'} {k}" for k, v in self.checks.items()]
        if self.functional_summary:
            lines.append(f"  ↳ test call: {self.functional_summary}")
        if self.warnings:
            lines.append("  ! " + "; ".join(self.warnings))
        if self.error:
            lines.append(f"  error: {self.error}")
        return "\n".join(lines)


@dataclass
class ToolDraft:
    """A generated candidate tool awaiting validation/approval."""

    name: str
    code: str
    spec: ToolGapSpec


# (draft, sandbox_result) -> approved?  Implemented by the CLI (shows code + result).
ApproveFn = Callable[[ToolDraft, SandboxResult], Awaitable[bool]]


_SYSTEM_PROMPT = """\
You author a single neurosurfer tool as Python source code.

A tool is a subclass of `neurosurfer.tools.base.Tool`:

    from pydantic import BaseModel, Field
    from neurosurfer.tools.base import Tool, ToolContext, ToolResult

    class <Name>Args(BaseModel):
        some_field: str = Field(description="...")

    class <Name>Tool(Tool):
        name = "<snake_case_name>"
        description = "<one concise sentence>"
        input_model = <Name>Args

        def is_read_only(self, args) -> bool:
            return True  # True if the tool only reads / computes; False if it writes

        async def call(self, args: <Name>Args, ctx: ToolContext) -> ToolResult:
            # ... do the work ...
            return ToolResult.ok("<result string>")   # or ToolResult.error("<why>")

Rules:
- Output EXACTLY ONE ```python code block and nothing else.
- Define exactly ONE Tool subclass; its `name` MUST equal the requested name.
- Prefer the standard library (pathlib, json, re, subprocess, ast, …). Keep it
  self-contained — no third-party packages unless unavoidable.
- `call` must be `async` and return a ToolResult.
- Be defensive: validate inputs, handle missing files/errors, return ToolResult.error
  with a clear message instead of raising.
- No network access unless the tool's purpose explicitly requires it.
"""


class ToolAuthor:
    """Generate, validate, and (with approval) register a tool for a capability gap."""

    def __init__(
        self,
        provider: Provider,
        *,
        cfg: GeneratedToolsConfig | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.provider = provider
        self.cfg = cfg or GeneratedToolsConfig()
        self.max_attempts = max_attempts

    async def author(
        self,
        spec: ToolGapSpec,
        *,
        approve: ApproveFn,
        notify: Callable[[str], None] | None = None,
    ) -> GeneratedToolMeta | None:
        """Author a tool for *spec*. Returns its meta if approved+registered, else None.

        ``approve`` is mandatory — there is no unattended registration path.
        On failure, the reason is recorded on ``self.last_failure`` (and surfaced via
        ``notify``) so callers can tell "couldn't generate" from "user rejected".
        """
        say = notify or (lambda _m: None)
        self.last_failure: str = ""
        self.rejected: bool = False
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            say(f"authoring '{spec.name}' (attempt {attempt}/{self.max_attempts})…")
            code = await self._generate(spec, last_error)
            if not code:
                last_error = "model returned no code block"
                say(f"  attempt {attempt}: {last_error}")
                continue
            draft = ToolDraft(name=spec.name, code=code, spec=spec)

            result = self.validate_draft(draft)
            if not result.ok:
                last_error = result.error or "validation failed"
                logger.info("tool author attempt %d failed: %s", attempt, last_error)
                say(f"  attempt {attempt} failed sandbox: {last_error}")
                continue

            # Passed validation — the human is the final gate.
            if not await approve(draft, result):
                logger.info("tool '%s' rejected by user", spec.name)
                self.rejected = True
                self.last_failure = "rejected by user"
                return None

            meta = GeneratedToolMeta(
                name=spec.name,
                description=spec.purpose,
                source_workflow=spec.source_workflow,
            )
            save_generated_tool(draft.code, meta, self.cfg)
            logger.info("tool '%s' authored and registered", spec.name)
            return meta

        self.last_failure = last_error or "could not produce a valid tool"
        return None

    # ── generation ───────────────────────────────────────────────────────────────
    async def _generate(self, spec: ToolGapSpec, last_error: str | None) -> str | None:
        prompt = self._build_prompt(spec, last_error)
        response = await self.provider.complete(
            messages=[Message.user_text(prompt)],
            system=_SYSTEM_PROMPT,
            tools=[],
            config=GenerationConfig(max_tokens=1500, temperature=0.3, stream=False),
        )
        return _extract_code_block(response.text())

    def _build_prompt(self, spec: ToolGapSpec, last_error: str | None) -> str:
        parts = [
            f"Write a tool named '{spec.name}'.",
            f"Purpose: {spec.purpose}",
        ]
        if spec.inputs:
            parts.append("Inputs it should accept:\n" + "\n".join(f"- {i}" for i in spec.inputs))
        if spec.context:
            parts.append(f"It will be used in this context: {spec.context}")
        if spec.expected_behavior:
            parts.append(f"On success it must: {spec.expected_behavior}")
        if spec.test_args or spec.test_setup:
            parts.append(
                "IMPORTANT: this tool will be FUNCTIONALLY TESTED — it is instantiated "
                "and `call()` is actually run in a sandbox. It must execute without "
                "raising and return a non-error ToolResult for these test arguments:\n"
                f"{spec.test_args}"
            )
            if spec.test_setup:
                parts.append(
                    "The sandbox first runs this setup to create fixtures in the working "
                    f"directory, so design call() to work against them:\n{spec.test_setup}"
                )
        if last_error:
            parts.append(
                "Your previous attempt failed validation with:\n"
                f"{last_error}\nFix it and output the corrected code only."
            )
        return "\n\n".join(parts)

    # ── validation: static scan + sandbox ────────────────────────────────────────
    def validate_draft(self, draft: ToolDraft) -> SandboxResult:
        checks: dict[str, bool] = {}
        warnings: list[str] = []

        # 1) Static: must parse.
        try:
            ast.parse(draft.code)
            checks["parses"] = True
        except SyntaxError as exc:
            return SandboxResult(ok=False, checks={"parses": False}, error=f"syntax error: {exc}")

        # 2) Static: surface risky tokens (flag, don't block).
        for pat in _RISKY_PATTERNS:
            if pat in draft.code:
                warnings.append(f"uses {pat!r}")

        # 3) Dynamic: import + instantiate + contract checks in a child process.
        sandbox = self._sandbox_check(draft)
        checks.update(sandbox.checks)
        warnings.extend(sandbox.warnings)

        ok = all(checks.values()) and not sandbox.error
        return SandboxResult(
            ok=ok,
            checks=checks,
            error=sandbox.error,
            warnings=warnings,
            functional_summary=sandbox.functional_summary,
        )

    def _sandbox_check(self, draft: ToolDraft) -> SandboxResult:
        """Import the candidate, instantiate it, assert its contract, and — when the
        spec supplies a test plan — actually CALL it once. All inside a
        timeout-bounded child process so untrusted code never runs in-process."""
        with tempfile.TemporaryDirectory(prefix="ma_toolcheck_") as tmp:
            tmpdir = Path(tmp)
            candidate = tmpdir / f"{draft.name}.py"
            candidate.write_text(draft.code, encoding="utf-8")
            runner = tmpdir / "_check.py"
            runner.write_text(_CHECK_RUNNER, encoding="utf-8")

            # Functional test plan (setup + args) for the child to run, if provided.
            spec_file = tmpdir / "_testspec.json"
            spec_file.write_text(
                json.dumps(
                    {"test_setup": draft.spec.test_setup, "test_args": draft.spec.test_args}
                ),
                encoding="utf-8",
            )

            env = dict(os.environ)
            # Replicate the parent's import paths so the child can import neurosurfer.
            env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

            try:
                proc = subprocess.run(
                    [sys.executable, str(runner), str(candidate), draft.name, str(spec_file)],
                    capture_output=True,
                    text=True,
                    timeout=_SANDBOX_TIMEOUT_S,
                    env=env,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    ok=False,
                    checks={"sandbox": False},
                    error=f"sandbox check timed out after {_SANDBOX_TIMEOUT_S}s",
                )

            out = (proc.stdout or "").strip()
            try:
                payload = json.loads(out.splitlines()[-1]) if out else {}
            except (json.JSONDecodeError, IndexError):
                return SandboxResult(
                    ok=False,
                    checks={"sandbox": False},
                    error=f"sandbox produced no result (stderr: {(proc.stderr or '').strip()[:300]})",
                )

            return SandboxResult(
                ok=bool(payload.get("ok")),
                checks=payload.get("checks", {}),
                error=payload.get("error", ""),
                functional_summary=payload.get("functional_summary", ""),
            )


# ── helpers ───────────────────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code_block(text: str) -> str | None:
    if not text:
        return None
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # No fences — accept raw source if it looks like a tool definition.
    if "class" in text and "Tool" in text and "def call" in text:
        return text.strip()
    return None


# Runner executed in the child process. Imports the candidate, finds the single Tool
# subclass, instantiates it, and verifies the contract. Prints one JSON line.
_CHECK_RUNNER = '''\
import sys, json, inspect, importlib.util

def main():
    path, expected = sys.argv[1], sys.argv[2]
    checks = {}
    try:
        spec = importlib.util.spec_from_file_location("_candidate_tool", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        checks["imports"] = True
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "checks": {"imports": False}, "error": f"import failed: {exc}"}))
        return

    try:
        from neurosurfer.tools.base import Tool
        from pydantic import BaseModel
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "checks": checks, "error": f"runtime import failed: {exc}"}))
        return

    tool_classes = [
        o for o in vars(mod).values()
        if inspect.isclass(o) and issubclass(o, Tool) and o is not Tool
        and o.__module__ == mod.__name__ and not inspect.isabstract(o)
    ]
    checks["single_tool_class"] = len(tool_classes) == 1
    if len(tool_classes) != 1:
        print(json.dumps({"ok": False, "checks": checks, "error": f"expected 1 Tool subclass, found {len(tool_classes)}"}))
        return

    cls = tool_classes[0]
    try:
        inst = cls()
        checks["instantiates"] = True
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "checks": checks, "error": f"instantiation failed: {exc}"}))
        return

    checks["name_matches"] = (getattr(inst, "name", None) == expected)
    checks["has_description"] = bool(getattr(inst, "description", ""))
    checks["input_model_ok"] = isinstance(getattr(inst, "input_model", None), type) and issubclass(inst.input_model, BaseModel)
    checks["call_is_async"] = inspect.iscoroutinefunction(getattr(cls, "call", None))

    try:
        _ = inst.schema  # exercises input_model -> json schema
        checks["schema_ok"] = True
    except Exception as exc:  # noqa: BLE001
        checks["schema_ok"] = False
        print(json.dumps({"ok": False, "checks": checks, "error": f"schema build failed: {exc}"}))
        return

    # ── Functional smoke test: actually CALL the tool once (when a plan is given) ──
    func_summary = ""
    spec_path = sys.argv[3] if len(sys.argv) > 3 else ""
    tspec = {}
    if spec_path:
        try:
            with open(spec_path) as fh:
                tspec = json.load(fh)
        except Exception:  # noqa: BLE001
            tspec = {}
    test_args = tspec.get("test_args") or {}
    setup = tspec.get("test_setup") or ""
    if test_args or setup:
        try:
            import asyncio, os
            from pathlib import Path
            ns = {}
            if setup:
                exec(compile(setup, "<test_setup>", "exec"), ns)
                if isinstance(ns.get("ARGS"), dict):
                    test_args = ns["ARGS"]

            from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext, ToolResult
            ctx = ToolContext(cwd=Path(os.getcwd()), io=AutoApproveIOHandler())
            args_obj = inst.input_model(**test_args)
            res = asyncio.run(inst.call(args_obj, ctx))
        except Exception as exc:  # noqa: BLE001
            checks["functional_runs"] = False
            print(json.dumps({"ok": False, "checks": checks, "error": f"functional test raised: {exc}"}))
            return
        if not isinstance(res, ToolResult):
            checks["functional_runs"] = False
            print(json.dumps({"ok": False, "checks": checks, "error": "call() did not return a ToolResult"}))
            return
        content = (getattr(res, "content", "") or "")[:200].replace(chr(10), " ")
        if getattr(res, "is_error", False):
            checks["functional_runs"] = False
            print(json.dumps({"ok": False, "checks": checks,
                              "error": "functional test returned an error result: " + content,
                              "functional_summary": "[error] " + content}))
            return
        checks["functional_runs"] = True
        func_summary = "[ok] " + content

    ok = all(checks.values())
    print(json.dumps({"ok": ok, "checks": checks, "error": "" if ok else "contract checks failed",
                      "functional_summary": func_summary}))


main()
'''
