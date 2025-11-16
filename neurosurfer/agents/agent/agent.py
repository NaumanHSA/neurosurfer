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

from ..common.utils import normalize_response
from .config import AgentConfig
from .templates import (
    TOOL_ROUTING_PROMPT,
    STRICT_TOOL_ROUTING_PROMPT,
    REPAIR_JSON_PROMPT,
    CORRECT_JSON_PROMPT,
)
from .schema_utils import (
    build_structured_system_prompt,
    extract_and_repair_json,
    maybe_unwrap_named_root,
)
from .responses import StructuredResponse, ToolCallResponse, AgentResponse
from neurosurfer.tracing import Tracer, TracerConfig


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
        Optional `Tracer` used to record and span steps in the agent flow.
    log_tracing:
        If True, tracer if enabled will log spans to the console per step.
        Only applicable if `tracer` is not None.

    Tracing behaviour
    -----------------
    When tracing is enabled, spans are opened around:
        * `agent.run`
        * `agent.structured_call`
        * `agent.route_and_call`
        * `agent.route_and_call.router_llm_call`
        * `agent.route_and_call.tool_execute`
        * `agent.free_text_call`
    Tracing results are available via the `traces` attribute of the `AgentResult` returned by `run()`, `structured_call()`, `route_and_call()`, and `free_text_call()`. 
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
        log_traces: Optional[bool] = True,
    ):
        self.llm = llm
        self.toolkit = toolkit
        self.config = config or AgentConfig()
        self.logger = logger
        self.verbose = verbose
        self.log_traces = log_traces

        # Tracing setup
        # Base tracer that actually records and log steps (RichTracer by default).
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "generic_agent",
                "agent_config": self.config,
                "model": llm.model_name,
                "toolkit": toolkit is not None,
                "verbose": verbose,
                "log_steps": self.log_traces,
            },
            logger_=logger,
        )

    # Public API to run the agent
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
        reset_tracer: bool = True,
    ) -> AgentResponse:
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

        Returns
        -------
        AgentResponse
            - `response`: Union[str, Generator[str, None, None], StructuredResponse, ToolCallResponse]
                - Free-text answer (string or streaming generator of strings),
                or
                - StructuredResponse (for Pydantic-validated outputs), or
                - ToolCallResponse (for tool routing mode).
            - `traces`: Tracing results for the run.
        """
        if reset_tracer:
            self.tracer.reset()   # Reset tracer before each run

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
        
        if output_schema and use_stream:
            self.logger.warning("`output_schema` provided with `stream=True`; forcing non-streaming structured output.")
            use_stream = False

        self._print("🧠 Thinking...", color="yellow")
        if self.log_traces:
            self._print("\nTracing Start!")
        with self.tracer.step(
            kind="agent",
            label="agent.run",
            inputs={
                "agent_type": type(self).__name__,
                "has_toolkit": bool(self.toolkit),
                "structured": bool(output_schema),
                "stream": use_stream,
                "strict_tool_call": strict_tool_call,
            },
        ):
            # Tool mode?
            if self.toolkit:
                response = self._route_and_call(
                    user_prompt=usr_prompt,
                    extra_instructions=_route_extra_instructions,
                    temperature=temp,
                    max_new_tokens=mnt,
                    use_stream=use_stream,
                    context=context or {},
                    strict_tool_call=strict_tool_call
                )
            # Structured mode via Pydantic
            elif output_schema is not None:
                response = self._structured_llm_call(
                    sys_prompt=sys_prompt,
                    usr_prompt=usr_prompt,
                    output_schema=output_schema,
                    temperature=temp,
                    max_new_tokens=mnt,
                )
            else:
                # Free-text mode
                llm_params = {
                    "user_prompt": usr_prompt,
                    "system_prompt": sys_prompt,
                    "temperature": temp,
                    "max_new_tokens": mnt,
                    "stream": use_stream,
                }
                with self.tracer.step(
                    kind="llm.call",
                    label="agent.free_text_call",
                    inputs={
                        "system_prompt_len": len(sys_prompt),
                        "user_prompt_len": len(usr_prompt),
                        **llm_params,
                    },
                ) as t:
                    response = normalize_response(self.llm.ask(**llm_params))
                    t.outputs(output=response)

        if self.log_traces:
            self._print("Tracing End!\n")  
        # Print final response
        if isinstance(response, str):
            self._print("Final response:", color="green")
            self._print(response.strip(), rich=False)
        return AgentResponse(
            response=response,
            traces=self.tracer.results,
        )

    # Structured output owned here (compact contract + parse + repair)
    def _structured_llm_call(
        self,
        *,
        sys_prompt: str,
        usr_prompt: str,
        output_schema: Type[PydModel],
        temperature: float,
        max_new_tokens: int,
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
       
        parsed_output = None
        sys, model_structure = build_structured_system_prompt(
            base_system_prompt=sys_prompt,
            schema_cls=output_schema,
            options=kwargs.pop("structured_prompt_options", None),
            use_model_json_schema=True,
        )
        llm_params = {"user_prompt": usr_prompt,"system_prompt": sys,"temperature": temperature,"max_new_tokens": max_new_tokens,"stream": False}
        # First pass: ask model for JSON directly
        with self.tracer.step(
            kind="llm.call",
            label="agent.structured_call.first_pass",
            inputs={
                "schema": output_schema.__name__,
                "system_prompt_len": len(sys_prompt),
                "user_prompt_len": len(usr_prompt),
                **llm_params,
            },
        )as t:
            model_response = self.llm.ask(**llm_params).choices[0].message.content or ""
            t.outputs(model_response=model_response, model_response_len=len(model_response))
            json_obj = extract_and_repair_json(model_response, return_dict=True)

        # Repair loop when we can't find JSON at all
        if json_obj is None and self.config.max_json_repair_attempts > 0:
            t = self.tracer.step(kind="llm.call", label="agent.structured_call.repair_json", inputs={"schema": output_schema.__name__}).start()
            try:
                t.log("Could not locate a JSON object in model output. Regenerating...", type="warning")
                repair_user_prompt = REPAIR_JSON_PROMPT.format(model_structure=model_structure, model_response=model_response)
                repair_sys_prompt = "You are a strict JSON fixer."
                llm_params = {"user_prompt": repair_user_prompt, "system_prompt": repair_sys_prompt, "temperature": 0.3, "max_new_tokens": max_new_tokens, "stream": False}
                t.inputs(system_prompt_len=len(repair_sys_prompt), user_prompt_len=len(repair_user_prompt), **llm_params)
                
                repaired_response = self.llm.ask(**llm_params).choices[0].message.content or ""
                t.outputs(repaired_json=repaired_response, repaired_json_len=len(repaired_response))
                json_obj = extract_and_repair_json(repaired_response, return_dict=True)
            except Exception as e:
                t.log(f"Structured output: could not locate a JSON object in model output with errors: {e}", type="error")
                json_obj = None
            t.close()

        if json_obj:
            json_obj = json.dumps(json_obj, ensure_ascii=False, indent=2)
            try:
                parsed_output = output_schema.model_validate_json(maybe_unwrap_named_root(json_obj, output_schema))
            except Exception as e:
                t = self.tracer.step(kind="llm.call", label="agent.structured_call.repair_validation", inputs={"schema": output_schema.__name__})
                # Validation failed -> retry repair once more via the model
                try:
                    repair_sys_prompt = "You are a strict JSON fixer."
                    repair_user_prompt = CORRECT_JSON_PROMPT.format(error=e, model_structure=model_structure, json_obj=json_obj)

                    llm_params = {"user_prompt": repair_user_prompt, "system_prompt": repair_sys_prompt, "temperature": 0.3, "max_new_tokens": max_new_tokens, "stream": False}
                    t.inputs(system_prompt_len=len(repair_sys_prompt), user_prompt_len=len(repair_user_prompt), **llm_params)
                    t.log("Structured output JSON failed validation: Retrying...", type="warning")
                    repaired_response = self.llm.ask(**llm_params).choices[0].message.content or ""
                    t.outputs(repaired_json=repaired_response, repaired_json_len=len(repaired_response))

                    # Try to extract and validate the repaired JSON
                    json_obj = extract_and_repair_json(repaired_response, return_dict=True)
                    json_obj = json.dumps(json_obj, ensure_ascii=False, indent=2)
                    parsed_output = output_schema.model_validate_json(maybe_unwrap_named_root(json_obj, output_schema))
                    t.log("Structured output JSON successfully validated...", type="info")
                except Exception as e:
                    json_obj = None
                    t.log(f"Structured output JSON failed validation with errors: {e}", type="error")
                t.close()

        return StructuredResponse(
            output_schema=output_schema,
            model_response=model_response,
            json_obj=json_obj,
            parsed_output=parsed_output,
        )

    # Tool routing + execution
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
    ) -> Union[str, ToolCallResponse]:
        """
        Route a request to a tool (or fall back to text) using the toolkit.

        This method:
          1. Builds a routing prompt including all tool descriptions.
          2. Asks the LLM to select a tool and provide inputs (JSON).
          3. Repairs invalid selections or inputs when possible.
          4. Executes the selected tool and normalizes the results.

        Tracing
        -------
        Opens a top-level span `agent.route_and_call`, plus:
          - `agent.route_and_call.router_llm_call`: for the LLM routing call.
          - `agent.route_and_call.tool_execute`: around the tool invocation.
        """
        tool_descriptions = self.toolkit.get_tools_description()
        router_prompt = (STRICT_TOOL_ROUTING_PROMPT if strict_tool_call else TOOL_ROUTING_PROMPT).format(
            tool_descriptions=tool_descriptions,
            extra_instructions=extra_instructions or "",
        )
        routing_attempt = 0
        repair_prompt = ""
        tool_name: str = ""
        tool_inputs: Dict[str, Any] = {}
        checked: Dict[str, Any] = {}

        t = self.tracer.step(
            kind="llm.call",
            label="agent.route_and_call.router_llm_call",
            inputs={
                "attempt": routing_attempt + 1,
                "strict_tool_call": strict_tool_call,
                "system_prompt_len": len(router_prompt),
                "user_prompt_len": len(user_prompt)
            },
        ).start()
        while True:
            user_prompt = user_prompt if not repair_prompt else f"{user_prompt}\n\n{repair_prompt}"
            llm_params = {
                "user_prompt": user_prompt,
                "system_prompt": router_prompt,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stream": False,
            }
            t.inputs(**llm_params)
            model_response = self.llm.ask(**llm_params).choices[0].message.content or ""
            t.outputs(model_response=model_response, model_response_len=len(model_response))

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
                    t.log("Returning plain response", type="info")
                    t.close()
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
                e_msg = (
                    "There was a problem selecting the right tool or repairing invalid "
                    "inputs while `strict_tool_call` is enabled."
                )
                t.log(e_msg, type="error")
                t.set_error(e_msg)
                return e_msg
            repair_prompt = (
                "[Previous issue]\nYour output was not valid JSON or did not select a "
                "usable tool. Return JSON only."
            )
            time.sleep(self.config.retry.backoff_sec * routing_attempt)

        t.log(f"Selected tool: {tool_name}")
        t.log(f"Raw inputs: {tool_inputs}")
        t.close()

        payload = {**checked, **(context or {})}
        tool = self.toolkit.registry.get(tool_name)

        with self.tracer.step(
            kind="tool.execute",
            label="agent.route_and_call.tool_execute",
            inputs={
                "tool_name": tool_name,
                "payload": payload,
            },
        ) as t:
            try:
                tool_response = tool(**payload)
                extras = tool_response.extras or {}
                tool_return = normalize_response(tool_response.results)
                t.outputs(tool_return=tool_return, extras=extras)
                t.log(f"Tool '{tool_name}' returned: {tool_return}", type="info")
                return ToolCallResponse(
                    selected_tool=tool_name,
                    inputs=checked,
                    returns=tool_return,
                    final=bool(tool_response.final_answer),
                    extras=extras,
                )
            except Exception as e:
                e_msg = f"Error while executing tool '{tool_name}': {e}.\nInputs: {checked}\n"
                t.set_error(e_msg)
                t.log(e_msg, type="error")
                return e_msg

    # Input validation for tools
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

    # Internal helper to print messages
    def _print(self, msg: str, color: str = "cyan", rich: bool = True):
        try:
            if rich:
                from rich.console import Console
                console = Console(force_jupyter=False, force_terminal=True)
                console.print(f"[underline][bold {color}]{msg}[/bold {color}]")
            else:
                print(msg)
        except NameError:
            print(msg)