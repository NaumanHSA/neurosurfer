# neurosurfer/tracing/workflow.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Literal
import time

class TraceStepContext:
    """
    Context manager returned by Tracer.step().

    - On __enter__:
        * records start time
        * optionally logs a span "▶ step.<kind>"
    - On __exit__:
        * sets duration_ms
        * sets ok/error
        * records a TraceStep via Tracer._record_step()
        * does NOT suppress exceptions
    """

    def __init__(
        self,
        *,
        tracer,
        step_id: int,
        kind: str,
        label: Optional[str],
        inputs: Dict[str, Any],
        agent_id: Optional[str],
        meta: Dict[str, Any],
    ) -> None:
        self._tracer = tracer
        self._step_data: Dict[str, Any] = {
            "step_id": step_id,
            "kind": kind,
            "label": label,
            "agent_id": agent_id,
            "inputs": inputs,
            "outputs": {},
            "meta": meta,
            "started_at": 0.0,
            "duration_ms": 0,
            "ok": True,
            "error": None,
            "logs": [],
        }
        self._span_cm = None
        self._is_closed = False

    def __enter__(self) -> "TraceStepContext":
        # increase nesting depth for this tracer
        self._tracer._depth += 1

        self._step_data["started_at"] = time.time()
        self._span_cm = self._tracer._span(
            name=f"step.{self._step_data['kind']}",
            attrs={
                "agent_id": self._step_data["agent_id"],
                "step_id": self._step_data["step_id"],
                "label": self._step_data["label"],
            },
        )
        self._span_cm.__enter__()
        return self

    def set_error(self, error: str) -> None:
        self._step_data["error"] = error
        self._step_data["ok"] = False

    def add_meta(self, **kwargs: Dict[str, Any]) -> None:
        self._step_data["meta"].update(kwargs)
    
    def outputs(self, **kwargs: Any) -> None:
        """
        Add arbitrary key/value pairs to `outputs`.

        Typical usage:
            t.outputs(output=normalized_result)
            t.outputs(system_prompt=sys, user_prompt=usr)
        """
        outputs: dict = self._step_data.setdefault("outputs", {})
        outputs.update(kwargs)

    def inputs(self, **kwargs: Any) -> None:
        """
        Add arbitrary key/value pairs to `inputs`.

        Typical usage:
            t.inputs(output=normalized_result)
            t.inputs(system_prompt=sys, user_prompt=usr)
        """
        inputs: dict = self._step_data.setdefault("inputs", {})
        inputs.update(kwargs)

    def stream(
        self, 
        message: str, 
        type: str = "info", 
        **data: Any
    ) -> None:
        """
        Add an internal log line to this step.

        - Stored in the structured trace result (step.logs)
        - Printed at the same indentation level as the step spans
          when `log_steps=True`.
        """
        try:
            indent_level = max(self._tracer._depth - 1, 0)
            self._tracer._log_line(
                step_id=self._step_data["step_id"],
                indent_level=indent_level,
                message=message,
                type=type,
                type_keyword=False,
                stream=True
            )
        except:
            pass

    def log(
        self, 
        message: str, 
        type: str = "info", 
        type_keyword: bool = True,
        **data: Any
    ) -> None:
        """
        Add an internal log line to this step.

        - Stored in the structured trace result (step.logs)
        - Printed at the same indentation level as the step spans
          when `log_steps=True`.
        """
        ts = time.time()
        # store in structured trace
        logs: list = self._step_data.setdefault("logs", [])
        logs.append(
            {
                "ts": ts,
                "message": message,
                "data": data or {},
                "type": type,
            }
        )
        # print as a log line (aligned with this step)
        indent_level = max(self._tracer._depth - 1, 0)
        self._tracer._log_line(
            step_id=self._step_data["step_id"],
            indent_level=indent_level,
            message=message,
            type=type,
            type_keyword=type_keyword,
            stream=False,
        )


    def start(self) -> "TraceStepContext":
        self.__enter__()
        return self

    def is_closed(self) -> bool:
        return self._is_closed

    def close(self) -> None:
        self._is_closed = True
        indent_level = max(self._tracer._depth - 1, 0)
        self._ensure_stream_newline(indent_level)
        self.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, tb) -> bool:
        end = time.time()
        self._step_data["duration_ms"] = int(
            (end - self._step_data["started_at"]) * 1000
        )
        if exc_type is not None:
            self._step_data["ok"] = False
            self._step_data["error"] = repr(exc)

        # end span
        self._span_cm.__exit__(exc_type, exc, tb)

        # decrease nesting depth
        self._tracer._depth -= 1

        # record step
        self._tracer._record_step(self._step_data)

        # do not suppress exceptions
        return False

    def _ensure_stream_newline(self, indent_level: int) -> None:
        """
        If this step used streaming logs, emit a final newline / blank line
        so the closing '◀ [step...]' appears on its own line.
        """

        if not hasattr(self._tracer, "_stream_started"):
            return

        if self._step_data["step_id"] not in self._tracer._stream_started:
            return

        # Emit one empty line with proper indentation (non-stream mode).
        self._tracer._log_line(
            step_id=self._step_data["step_id"],
            indent_level=indent_level,
            message="\n",          # empty message => just indent + newline
            type="info",
            type_keyword=False,
            stream=False,
        )

        # Clean up streaming state for this step
        self._tracer._stream_started.discard(self._step_data["step_id"])
        if hasattr(self._tracer, "_stream_at_line_start"):
            self._tracer._stream_at_line_start.pop(self._step_data["step_id"], None)