from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, Generator, List, Optional, Union, Type

from pydantic import BaseModel as PydModel

from neurosurfer.models.chat_models.base import BaseModel as ChatBaseModel
from neurosurfer.server.schemas import ChatCompletionChunk, ChatCompletionResponse
from neurosurfer.tools import Toolkit
from neurosurfer.tools.tool_spec import TOOL_TYPE_CAST

from ..common.utils import normalize_tool_observation
from .config import AgentConfig
from .templates import TOOL_ROUTING_PROMPT, STRICT_TOOL_ROUTING_PROMPT
from .schema_utils import (
    build_structured_system_prompt,
    extract_and_repair_json,
    maybe_unwrap_named_root,
)
from .responses import StructuredResponse, ToolCallResponse
from neurosurfer.agents.common.tracing import Tracer, NullTracer, LoggerTracer, RichTracer


class Agent:
    """
    Generic LLM Agent with optional tool-calling and structured outputs.

    This Agent is responsible for:
      - Plain LLM calls (free-text responses).
      - Structured JSON outputs validated against Pydantic models.
      - Tool routing and execution when a `Toolkit` is provided.
      - JSON prompting / parsing / repair (keeps ChatBaseModel slim).
      - Optional tracing of key steps in the agentic flow via a pluggable `Tracer`.

    Parameters
    ----------
    llm:
        Concrete chat model implementing `ChatBaseModel`. The agent calls this
        for both routing and final answers.
    toolkit:
        Optional `Toolkit` to enable tool routing + execution. If provided,
        `run()` will attempt to route to a tool by default.
    config:
        Agent configuration object (see `AgentConfig`), including retry and
        JSON repair behaviour.
    logger:
        Python logger used for diagnostic messages.
    verbose:
        If True, the agent logs more detail (e.g. selected tool, errors).
    tracer:
        Optional `Tracer` implementation used to record spans. If None, a
        `LoggerTracer` based on `logger` is used as the base tracer.
    enable_tracing:
        If True, tracing is enabled by default for all `run()` calls.
        If False, tracing is disabled by default (but can still be enabled
        per-call via `trace=True` in `run()`).
        If None, defaults to the value of `verbose`.

    Tracing behaviour
    -----------------
    - The agent uses a small helper `_get_tracer(trace)` in `run()` to choose
      the effective tracer for a given call:
        * `trace` argument to `run()` (if provided) overrides everything.
        * Otherwise it falls back to `enable_tracing` or `verbose`.
    - When tracing is disabled for a call, a `NullTracer` is used, which is a
      drop-in tracer that does nothing (zero overhead in your flow).
    - Spans are opened around:
        * `agent.run`
        * `agent.structured_call`
        * `agent.route_and_call`
        * `agent.route_and_call.router_llm_call`
        * `agent.route_and_call.tool_execute`
        * `agent.free_text_call`
    """

    def __init__(
        self,
        llm: ChatBaseModel,
        toolkit: Optional[Toolkit] = None,
        *,
        config: Optional[AgentConfig] = None,
        logger: logging.Logger = logging.getLogger(__name__),
        verbose: bool = False,
        tracer: Optional[Tracer] = None,
        enable_tracing: Optional[bool] = None,
    ):
        self.llm = llm
        self.toolkit = toolkit
        self.config = config or AgentConfig()
        self.logger = logger
        self.verbose = verbose

        # Tracing setup
        # -------------
        # Base tracer that actually records spans (LoggerTracer by default).
        self._base_tracer: Tracer = tracer or RichTracer()
        # Null tracer to cheaply disable tracing.
        self._null_tracer: Tracer = NullTracer()
        # Default "is tracing enabled?" behaviour.
        self._enable_tracing_default: bool = (
            bool(enable_tracing) if enable_tracing is not None else bool(verbose)
        )
        # Expose base tracer for external use (e.g. graph nodes).
        self.tracer: Tracer = self._base_tracer

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(
        self,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        query: Optional[str] = None,
        output_schema: Optional[Type[PydModel]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        _route_extra_instructions: str = "",
        strict_tool_call: Optional[bool] = None,
        trace: Optional[bool] = None,
    ) -> Union[str, Generator[str, None, None], StructuredResponse, ToolCallResponse]:
        """
        Run a single agent step.

        Depending on the configuration and parameters, this will:
          - Call the LLM directly and return a free-text answer; OR
          - Use structured output with a Pydantic schema; OR
          - Route to a tool via the toolkit and return a `ToolCallResponse`.

        Parameters
        ----------
        system_prompt:
            Optional system prompt for the LLM. If omitted, a simple default
            "helpful assistant" prompt is used.
        user_prompt:
            Main user query / instruction. If omitted, `query` is used.
        query:
            Alias for `user_prompt`, mainly for backwards compatibility.
        output_schema:
            Optional Pydantic model class. If provided and no toolkit is set,
            the agent will attempt a structured JSON response, validate it
            against this model, and return a `StructuredResponse`.
        stream:
            If True, requests streaming from the underlying LLM. If None,
            falls back to `config.return_stream_by_default`.
        temperature:
            Sampling temperature for the LLM. If None, uses `config.temperature`.
        max_new_tokens:
            Maximum number of new tokens to generate. If None, uses
            `config.max_new_tokens`.
        context:
            Additional context dictionary merged into tool inputs when
            executing a tool. Ignored for free-text and structured calls.
        _route_extra_instructions:
            Additional routing instructions appended to the system prompt for
            tool selection. Mostly internal.
        strict_tool_call:
            If True, the router must select a valid tool and repair invalid
            inputs; free-text fallback is disabled. If None, falls back to
            `config.strict_tool_call`.
        trace:
            Optional per-call override for tracing:
              - None: use default from `enable_tracing` / `verbose`.
              - True: force tracing on for this call.
              - False: force tracing off for this call.

        Returns
        -------
        Union[str, Generator[str, None, None], StructuredResponse, ToolCallResponse]
            - Free-text answer (string or streaming generator of strings),
              or
            - StructuredResponse (for Pydantic-validated outputs), or
            - ToolCallResponse (for tool routing mode).
        """
        tracer = self._get_tracer(trace)

        use_stream = self.config.return_stream_by_default if stream is None else bool(stream)
        temp = float(self.config.temperature if temperature is None else temperature)
        mnt = int(self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        sys_prompt = system_prompt or "You are a helpful assistant. Be concise and precise."
        usr_prompt = (user_prompt if user_prompt is not None else query) or ""
        strict_tool_call = self.config.strict_tool_call if strict_tool_call is None else bool(strict_tool_call)

        if self.toolkit and output_schema:
            self.logger.warning(
                "Structured responses are not supported when using a toolkit. "
                "Responses are returned from tools in case of a tool call. Ignoring output schema."
            )
            output_schema = None

        with tracer.span(
            "agent.run",
            {
                "agent_type": type(self).__name__,
                "has_toolkit": bool(self.toolkit),
                "structured": bool(output_schema),
                "stream": use_stream,
                "strict_tool_call": strict_tool_call,
            },
        ):
            self.logger.info("🧠 Thinking...")

            # Tool mode?
            if self.toolkit:
                return self._route_and_call(
                    user_prompt=usr_prompt,
                    extra_instructions=_route_extra_instructions,
                    temperature=temp,
                    max_new_tokens=mnt,
                    use_stream=use_stream,
                    context=context or {},
                    strict_tool_call=strict_tool_call,
                    tracer=tracer,
                )

            # Structured mode via Pydantic
            if output_schema is not None:
                if use_stream:
                    self.logger.warning(
                        "`output_schema` provided with `stream=True`; "
                        "forcing non-streaming structured output."
                    )
                return self._structured_llm_call(
                    sys_prompt=sys_prompt,
                    usr_prompt=usr_prompt,
                    output_schema=output_schema,
                    temperature=temp,
                    max_new_tokens=mnt,
                    tracer=tracer,
                )

            # Free-text mode
            with tracer.span(
                "agent.free_text_call",
                {
                    "system_prompt_len": len(sys_prompt),
                    "user_prompt_len": len(usr_prompt),
                    "stream": use_stream,
                },
            ):
                response = self.llm.ask(
                    user_prompt=usr_prompt,
                    system_prompt=sys_prompt,
                    temperature=temp,
                    max_new_tokens=mnt,
                    stream=use_stream,
                )
            return normalize_tool_observation(response)

    # --------------------------------------------------------------------- #
    # Structured output owned here (compact contract + parse + repair)
    # --------------------------------------------------------------------- #
    def _structured_llm_call(
        self,
        *,
        sys_prompt: str,
        usr_prompt: str,
        output_schema: Type[PydModel],
        temperature: float,
        max_new_tokens: int,
        tracer: Tracer,
        **kwargs: Any,
    ) -> StructuredResponse:
        """
        Perform a structured LLM call that returns JSON, repair it if needed,
        and validate against a Pydantic model.

        This method:
          1. Builds a compact structure description from `output_schema`.
          2. Calls the LLM to generate JSON-like output.
          3. Extracts and repairs JSON using `extract_and_repair_json`.
          4. Validates the final JSON against the Pydantic model.
          5. Optionally retries repair if validation fails.

        Tracing
        -------
        Opens a span `agent.structured_call` with attributes including the
        schema name and token limits.
        """
        with tracer.span(
            "agent.structured_call",
            {
                "schema": output_schema.__name__,
                "system_prompt_len": len(sys_prompt),
                "user_prompt_len": len(usr_prompt),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        ):
            parsed_output = None
            sys, model_structure = build_structured_system_prompt(
                base_system_prompt=sys_prompt,
                schema_cls=output_schema,
                options=kwargs.pop("structured_prompt_options", None),
                use_model_json_schema=True,
            )

            # First pass: ask model for JSON directly
            with tracer.span("llm.structured_call", {"schema": output_schema.__name__}):
                model_response = (
                    self.llm.ask(
                        user_prompt=usr_prompt,
                        system_prompt=sys,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        stream=False,
                        **kwargs,
                    )
                    .choices[0]
                    .message.content
                    or ""
                )
            json_obj = extract_and_repair_json(model_response, return_dict=True)
            # Repair loop when we can't find JSON at all
            if json_obj is None and self.config.max_repair_attempts > 0:
                try:
                    self.logger.warning(
                        "Could not locate a JSON object in model output. Regenerating..."
                    )
                    with tracer.span("llm.structured_repair_no_json", {"attempt": 1}):
                        repaired_response = (
                            self.llm.ask(
                                user_prompt=(
                                    "Fix the following model output into valid JSON that matches the structure.\n\n"
                                    f"### Structure\n{model_structure}\n\n"
                                    "### Output to fix\n"
                                    f"{model_response}\n\n"
                                    "Return ONLY the JSON object. No markdown, no comments."
                                ),
                                system_prompt="You are a strict JSON fixer.",
                                temperature=0.3,
                                max_new_tokens=max_new_tokens,
                                stream=False,
                            )
                            .choices[0]
                            .message.content
                            or ""
                        )
                    json_obj = extract_and_repair_json(repaired_response, return_dict=True)
                except Exception as e:
                    self.logger.error(
                        f"Structured output: could not locate a JSON object in model output with errors: {e}"
                    )
                    json_obj = None

            if json_obj:
                json_obj = json.dumps(json_obj, ensure_ascii=False, indent=2)
                try:
                    parsed_output = output_schema.model_validate_json(
                        maybe_unwrap_named_root(json_obj, output_schema)
                    )
                except Exception as e:
                    # Validation failed -> retry repair once more via the model
                    self.logger.warning(
                        "Structured output JSON failed validation: Retrying..."
                    )
                    try:
                        with tracer.span(
                            "llm.structured_repair_validation",
                            {"schema": output_schema.__name__},
                        ):
                            repaired_response = (
                                self.llm.ask(
                                    user_prompt=(
                                        "Parsing JSON to Pydantic Model failed with error: "
                                        + str(e)
                                        + "\n\n"
                                        "Fix the following model output into valid JSON that matches the structure.\n\n"
                                        f"### Structure\n{model_structure}\n\n"
                                        "### Output to fix\n"
                                        f"{json_obj}\n\n"
                                        "Return ONLY the JSON object. No markdown, no comments."
                                    ),
                                    system_prompt="You are a strict JSON fixer.",
                                    temperature=0.3,
                                    max_new_tokens=max_new_tokens,
                                    stream=False,
                                )
                                .choices[0]
                                .message.content
                                or ""
                            )
                        json_obj = extract_and_repair_json(repaired_response, return_dict=True)
                        json_obj = json.dumps(json_obj, ensure_ascii=False, indent=2)
                        parsed_output = output_schema.model_validate_json(
                            maybe_unwrap_named_root(json_obj, output_schema)
                        )
                        self.logger.info("Structured output JSON successfully validated...")
                    except Exception:
                        json_obj = None
                        self.logger.error(
                            f"Structured output JSON failed validation with errors: {e}"
                        )

            return StructuredResponse(
                output_schema=output_schema,
                model_response=model_response,
                json_obj=json_obj,
                parsed_output=parsed_output,
            )

    # --------------------------------------------------------------------- #
    # Tool routing + execution
    # --------------------------------------------------------------------- #
    def _route_and_call(
        self,
        *,
        user_prompt: str,
        extra_instructions: str,
        temperature: float,
        max_new_tokens: int,
        use_stream: bool,
        context: Dict[str, Any],
        strict_tool_call: bool,
        tracer: Tracer,
    ) -> Union[str, ToolCallResponse]:
        """
        Route a request to a tool (or fall back to text) using the toolkit.

        This method:
          1. Builds a routing prompt including all tool descriptions.
          2. Asks the LLM to select a tool and provide inputs (JSON).
          3. Repairs invalid selections or inputs when possible.
          4. Executes the selected tool and normalizes the observation.

        Tracing
        -------
        Opens a top-level span `agent.route_and_call`, plus:
          - `agent.route_and_call.router_llm_call`: for the LLM routing call.
          - `agent.route_and_call.tool_execute`: around the tool invocation.
        """
        tool_descriptions = self.toolkit.get_tools_description()
        router_prompt = (
            STRICT_TOOL_ROUTING_PROMPT if strict_tool_call else TOOL_ROUTING_PROMPT
        ).format(
            tool_descriptions=tool_descriptions,
            extra_instructions=extra_instructions or "",
        )

        routing_attempt = 0
        repair_prompt = ""
        tool_name: str = ""
        tool_inputs: Dict[str, Any] = {}
        checked: Dict[str, Any] = {}

        with tracer.span(
            "agent.route_and_call",
            {
                "user_prompt_len": len(user_prompt),
                "strict_tool_call": strict_tool_call,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        ):
            while True:
                with tracer.span(
                    "agent.route_and_call.router_llm_call",
                    {
                        "attempt": routing_attempt + 1,
                        "strict_tool_call": strict_tool_call,
                    },
                ):
                    model_response = (
                        self.llm.ask(
                            user_prompt=(
                                user_prompt
                                if not repair_prompt
                                else f"{user_prompt}\n\n{repair_prompt}"
                            ),
                            system_prompt=router_prompt,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            stream=False,
                        )
                        .choices[0]
                        .message.content
                        or ""
                    )

                # try parsing json, in case model suggests a tool call
                obj = extract_and_repair_json(model_response, return_dict=True)
                if obj and isinstance(obj, dict) and "tool" in obj and "inputs" in obj:
                    tool_name = str(obj["tool"])
                    tool_inputs = obj["inputs"] if isinstance(obj["inputs"], dict) else {}
                else:
                    if strict_tool_call:
                        tool_name, tool_inputs = "none", {}
                    else:
                        # return plain response if model can choose between tool and direct answer
                        return model_response

                # for strict tool calling, validate tool call and repair inputs if invalid
                if tool_name and tool_name != "none":
                    tool = self.toolkit.registry.get(tool_name)
                    if tool is None:
                        repair_prompt = (
                            f"[Previous issue]\nThe tool '{tool_name}' you selected is not "
                            "registered with the toolkit. Please select the right tool, "
                            "double-check the name."
                        )
                        routing_attempt += 1
                        continue
                    try:
                        checked = self._validate_inputs(tool_name, tool_inputs)
                    except Exception as e:
                        repair_prompt = (
                            f"[Previous issue]\nFix inputs for tool '{tool_name}'. Error: {e}\n"
                            f"Current: {tool_inputs}\n"
                            "Return ONLY a JSON object for 'inputs'. Please double-check the input "
                            "names and types."
                        )
                        routing_attempt += 1
                        continue
                    break

                # If we get here, no usable tool was selected; try repair or give up
                routing_attempt += 1
                if (
                    routing_attempt > self.config.retry.max_route_retries
                    or not self.config.repair_with_llm
                ):
                    return (
                        "There was a problem selecting the right tool or repairing invalid "
                        "inputs while `strict_tool_call` is enabled."
                    )

                repair_prompt = (
                    "[Previous issue]\nYour output was not valid JSON or did not select a "
                    "usable tool. Return JSON only."
                )
                time.sleep(self.config.retry.backoff_sec * routing_attempt)

            if self.verbose:
                self.logger.info(f"[ToolsCallingAgent] Selected tool: {tool_name}")
                self.logger.info(f"[ToolsCallingAgent] Raw inputs: {tool_inputs}")

            payload = {**checked, **(context or {})}
            tool = self.toolkit.registry.get(tool_name)

            with tracer.span(
                "agent.route_and_call.tool_execute",
                {"tool_name": tool_name, "payload_keys": list(payload.keys())},
            ):
                try:
                    tool_response = tool(**payload)
                    extras = tool_response.extras or {}
                    tool_return = normalize_tool_observation(tool_response.observation)
                    return ToolCallResponse(
                        selected_tool=tool_name,
                        inputs=checked,
                        returns=tool_return,
                        final=bool(tool_response.final_answer),
                        extras=extras,
                    )
                except Exception as e:
                    if self.verbose:
                        self.logger.exception(
                            f"[ToolsCallingAgent] Tool '{tool_name}' error: {e}"
                        )
                    return (
                        f"Error while executing tool '{tool_name}': {e}.\n"
                        f"Inputs: {checked}\n"
                    )

    # --------------------------------------------------------------------- #
    # Input validation for tools
    # --------------------------------------------------------------------- #
    def _validate_inputs(self, tool_name: str, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and cast raw tool inputs according to the tool's `ToolSpec`.

        - Optionally prunes extra keys (if `config.allow_input_pruning` is True).
        - Casts values using `TOOL_TYPE_CAST` when types are declared.
        - Relies on `tool.spec.check_inputs` for final validation.
        """
        tool = self.toolkit.registry.get(tool_name)
        if tool is None:
            return raw_inputs or {}

        inputs = dict(raw_inputs or {})

        if self.config.allow_input_pruning:
            allowed = {p.name for p in tool.spec.inputs}
            inputs = {k: v for k, v in inputs.items() if k in allowed}

        for p in tool.spec.inputs:
            if p.name in inputs and p.type in TOOL_TYPE_CAST:
                try:
                    inputs[p.name] = TOOL_TYPE_CAST[p.type](inputs[p.name])
                except Exception as e:
                    raise ValueError(f"Invalid input for parameter '{p.name}': {e}")

        return tool.spec.check_inputs(inputs)

    # --------------------------------------------------------------------- #
    # Internal helper
    # --------------------------------------------------------------------- #
    def _get_tracer(self, trace: Optional[bool]) -> Tracer:
        """
        Resolve the effective tracer for a single `run()` call.

        - If `trace` is explicitly set on the call, we respect that.
        - Otherwise fall back to `_enable_tracing_default` which is derived
          from `enable_tracing` (if provided) or `verbose`.
        """
        effective = self._enable_tracing_default if trace is None else bool(trace)
        return self._base_tracer if effective else self._null_tracer
