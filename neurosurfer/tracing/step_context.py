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

    def log(self, message: str, type: Literal["info", "warning", "error", "debug"] = "info", **data: Any) -> None:
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
        )

    def start(self) -> "TraceStepContext":
        self.__enter__()
        return self

    def close(self) -> None:
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