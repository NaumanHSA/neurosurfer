from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple, Union
import math

from .models import TraceResult, TraceStep
from .config import TracerConfig


@dataclass
class PrettyTraceConfig:
    format: Literal["text", "markdown"] = "text"

    indent_spaces: int = 4
    show_agent_banner: bool = True
    show_synthetic_phase_lines: bool = True

    show_inputs: bool = False
    show_outputs: bool = False
    max_preview_chars: int = 300

    show_logs: bool = True
    include_log_data: bool = False  # include log["data"] dict after message if present

    # cosmetic
    arrow_open: str = "▶"
    arrow_close: str = "◀"

def render_trace_result(trace: Union[TraceResult, Dict[str, Any]], cfg: Optional[TracerConfig] = None, format: Literal["text", "markdown"] = "text") -> str:
    """
    Convert TraceResult into a pretty, nested log string (optionally Markdown).

    - Uses timestamps to reconstruct nesting (correct even if step_id ordering is not nested).
    - Inserts step.logs lines at their recorded timestamps.
    """
    cfg = cfg or TracerConfig()
    if isinstance(trace, dict):
        trace = TraceResult(**trace)

    events = _build_timeline_events(trace, cfg)
    lines: List[str] = []

    if format == "markdown":
        lines.append("# Execution Trace\n")
        lines.append("```")
        lines.extend(_render_events(events, cfg))
        lines.append("```")
        return "\n".join(lines)

    return "\n".join(_render_events(events, cfg))


@dataclass(frozen=True)
class _Event:
    t: float
    kind: Literal["start", "end", "log"]
    step: Optional[TraceStep] = None
    log: Optional[Dict[str, Any]] = None


def _build_timeline_events(trace: TraceResult, cfg: TracerConfig) -> List[_Event]:
    events: List[_Event] = []

    for s in trace.steps:
        if s.started_at is None:
            continue

        start_t = float(s.started_at)
        dur_s = (float(s.duration_ms) / 1000.0) if s.duration_ms is not None else 0.0
        end_t = start_t + max(dur_s, 0.0)

        events.append(_Event(t=start_t, kind="start", step=s))
        events.append(_Event(t=end_t, kind="end", step=s))

        if cfg.show_logs and getattr(s, "logs", None):
            for lg in s.logs:
                if lg.ts is None:
                    continue
                events.append(_Event(t=float(lg.ts), kind="log", step=s, log=lg.model_dump()))

    # Sort rules:
    # - Earlier timestamps first
    # - For equal timestamps: start -> log -> end (so we don't close before printing logs)
    order = {"start": 0, "log": 1, "end": 2}
    events.sort(key=lambda e: (e.t, order[e.kind], getattr(e.step, "step_id", 0) if e.step else 0))
    return events

def _render_events(events: List[_Event], cfg: TracerConfig) -> List[str]:
    lines: List[str] = []
    stack: List[TraceStep] = []
    active_ids: set[int] = set()

    # Keep a small helper to compute indent
    def ind(level: int) -> str:
        return " " * (cfg.indent_spaces * max(level, 0))

    def fmt_step_open(level: int, s: TraceStep) -> str:
        return (
            f"{ind(level)}{cfg.arrow_open} "
            f"[{s.step_id}][step.{s.kind}] "
            f"agent_id='{s.agent_id}' label='{s.label}'"
        )

    def fmt_step_close(level: int, s: TraceStep) -> str:
        took = f"{(s.duration_ms or 0)/1000.0:.3f}s"
        err = "True" if (not getattr(s, "ok", True)) or getattr(s, "error", None) else "False"
        return (
            f"{ind(level)}{cfg.arrow_close} "
            f"[{s.step_id}][step.{s.kind}] "
            f"agent_id='{s.agent_id}' label='{s.label}' took {took}; error={err}"
        )

    def maybe_banner_start(level: int, s: TraceStep) -> None:
        if not cfg.show_agent_banner:
            return

        # Graph start banner
        if s.kind == "graph.execute":
            lines.append(f"[{s.agent_id}] Executing Graph...")

        if not cfg.show_synthetic_phase_lines:
            return

        # Manager banners
        if s.agent_id == "manager" and s.label == "manager.compose_user_prompt":
            lines.append(f"{ind(level)}[manager] Preparing Next Node...")

        # Node banners
        if s.label in ("agent.run", "react_agent.run"):
            lines.append(f"{ind(level)}[{s.agent_id}] 🧠 Thinking...")

        # LLM call banners (optional, but matches your sample)
        if s.kind == "llm.call":
            lines.append(f"{ind(level)}[{s.agent_id}] Starting llm.call...")

    def maybe_banner_end(level: int, s: TraceStep) -> None:
        if not cfg.show_synthetic_phase_lines and not cfg.show_agent_banner:
            return

        if cfg.show_synthetic_phase_lines:
            if s.kind == "llm.call":
                lines.append(f"{ind(level)}[{s.agent_id}] Completed llm.call!")

            if s.label in ("agent.run", "react_agent.run"):
                lines.append(f"{ind(level)}[{s.agent_id}] 🧠 Done!")

            if s.agent_id == "manager" and s.label == "manager.compose_user_prompt":
                lines.append(f"{ind(level)}[manager] Next Node Ready For Execution!")

        if cfg.show_agent_banner and s.kind == "graph.execute":
            lines.append(f"[{s.agent_id}] Graph Execution Complete!")

    def render_log(level: int, e: _Event) -> None:
        lg = e.log or {}
        msg = lg.get("message", "")
        typ = lg.get("type", "info")

        # Keep it close to your target style:
        prefix = typ.upper() + ": " if typ else ""
        line = f"{ind(level)}{prefix}{msg}"

        if cfg.include_log_data:
            data = lg.get("data")
            if data:
                line += f" {data}"

        lines.append(line)

    # Main replay loop
    for e in events:
        if e.kind == "start" and e.step is not None:
            s = e.step
            level = len(stack)  # indent BEFORE push

            maybe_banner_start(level, s)
            lines.append(fmt_step_open(level, s))

            stack.append(s)
            active_ids.add(s.step_id)

            if cfg.show_inputs and getattr(s, "inputs", None):
                inp = _preview_dict(s.inputs, cfg.max_preview_chars)
                lines.append(f"{ind(level+1)}inputs: {inp}")

            continue

        if e.kind == "log" and e.step is not None:
            # logs should appear under current stack depth
            level = len(stack)
            render_log(level, e)
            continue

        if e.kind == "end" and e.step is not None:
            s = e.step

            # Pop until we find this step (defensive, handles slight ordering quirks)
            if s.step_id in active_ids:
                while stack and stack[-1].step_id != s.step_id:
                    stray = stack.pop()
                    active_ids.discard(stray.step_id)
                if stack and stack[-1].step_id == s.step_id:
                    stack.pop()
                    active_ids.discard(s.step_id)

            level = len(stack)  # indent AFTER pop
            lines.append(fmt_step_close(level, s))

            if cfg.show_outputs and getattr(s, "outputs", None):
                out = _preview_dict(s.outputs, cfg.max_preview_chars)
                lines.append(f"{ind(level+1)}outputs: {out}")

            maybe_banner_end(level, s)
            continue

    return lines


def _preview_dict(d: Any, max_chars: int) -> str:
    try:
        s = str(d)
    except Exception:
        s = "<unprintable>"
    if len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s
