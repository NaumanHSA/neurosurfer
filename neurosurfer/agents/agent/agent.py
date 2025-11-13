from __future__ import annotations

import json, time, logging
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union, Type

from pydantic import BaseModel as PydModel, ValidationError

from neurosurfer.models.chat_models.base import BaseModel as ChatBaseModel
from neurosurfer.server.schemas import ChatCompletionChunk, ChatCompletionResponse
from neurosurfer.tools import Toolkit
from neurosurfer.tools.tool_spec import ToolSpec, TOOL_TYPE_CAST

from ..common.utils import normalize_tool_observation
from .config import AgentConfig
from .templates import TOOL_ROUTING_PROMPT, STRICT_TOOL_ROUTING_PROMPT
from .schema_utils import build_structured_system_prompt, extract_and_repair_json, maybe_unwrap_named_root
from .responses import StructuredResponse, ToolCallResponse


class Agent:
    """
    Agent that:
      - Plain LLM calls (text or structured)
      - Or tool-calling (router) when `tools` provided
      - Structured output can be from a List[str] spec OR a Pydantic model
      - Owns JSON prompting/parsing/repair (BaseModel stays slim)
    """

    def __init__(
        self,
        llm: ChatBaseModel,
        toolkit: Optional[Toolkit] = None,
        *,
        config: Optional[AgentConfig] = None,
        logger: logging.Logger = logging.getLogger(__name__),
        verbose: bool = False,
    ):
        self.llm = llm
        self.toolkit = toolkit
        self.config = config or AgentConfig()
        self.logger = logger
        self.verbose = verbose

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(
        self,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        query: Optional[str] = None,
        output_schema: Optional[PydModel] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        _route_extra_instructions: str = "",
        strict_tool_call: Optional[bool] = None,
    ) -> Union[str, Generator[str, None, None], StructuredResponse, ToolCallResponse]:

        use_stream = self.config.return_stream_by_default if stream is None else bool(stream)
        temp = float(self.config.temperature if temperature is None else temperature)
        mnt = int(self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        sys_prompt = system_prompt or "You are a helpful assistant. Be concise and precise."
        usr_prompt = (user_prompt if user_prompt is not None else query) or ""
        strict_tool_call = self.config.strict_tool_call if strict_tool_call is None else bool(strict_tool_call)

        if self.toolkit and output_schema:
            self.logger.warning("Structured responses are not supported when using a toolkit. Responses are returned from tools in case of a tool call. Ignoring output schema.")
            output_schema = None

        self.logger.info(f"🧠 Thinking...")
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
            )
        # Structured mode via Pydantic
        if output_schema is not None:
            if use_stream:
                self.logger.warning("`output_schema` provided with `stream=True`; forcing non-streaming structured output.")
            return self._structured_llm_call(
                sys_prompt=sys_prompt,
                usr_prompt=usr_prompt,
                output_schema=output_schema,
                temperature=temp,
                max_new_tokens=mnt,
            )
        # Free-text mode
        response = self.llm.ask(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            temperature=temp,
            max_new_tokens=mnt,
            stream=use_stream,
        )
        return normalize_tool_observation(response)

    # Structured output owned here (compact contract + parse + repair)
    def _structured_llm_call(
        self,
        *,
        sys_prompt: str,
        usr_prompt: str,
        output_schema: PydModel,
        temperature: float,
        max_new_tokens: int,
        **kwargs,
    ) -> StructuredResponse:
        # Build a tiny, human-readable structure prompt (no huge JSON schema)
        parsed_output = None
        sys, model_structure = build_structured_system_prompt(
            base_system_prompt=sys_prompt,
            schema_cls=output_schema,
            options=kwargs.pop("structured_prompt_options", None),
            use_model_json_schema=True,
        )
        model_response = self.llm.ask(
            user_prompt=usr_prompt,
            system_prompt=sys,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=False,
            **kwargs,
        ).choices[0].message.content or ""

        # assuming model returns json, extract and repair if needed
        json_obj = extract_and_repair_json(model_response, return_dict=True)
        if json_obj is None and self.config.max_repair_attempts > 0:
            try:
                self.logger.warning("Could not locate a JSON object in model output. Regenerating...")
                repaired_response = self.llm.ask(
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
                ).choices[0].message.content or ""
                json_obj = extract_and_repair_json(repaired_response, return_dict=True)
            except Exception as e:
                self.logger.error(f"Structured output: could not locate a JSON object in model output with errors: {e}")
                json_obj = None

        if json_obj:
            json_obj = json.dumps(json_obj, ensure_ascii=False, indent=2)
            # Validate against your real Pydantic model (no content shaping)
            try:
                parsed_output = output_schema.model_validate_json(maybe_unwrap_named_root(json_obj, output_schema))
            except Exception as e:
                # retry regenerating
                self.logger.warning(f"Structured output JSON failed validation: Retrying...")
                try:
                    repaired_response = self.llm.ask(
                        user_prompt=(
                            "Parsing JSON to Pydantic Model failed with error: " + str(e) + "\n\n"
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
                    ).choices[0].message.content or ""
                    json_obj = extract_and_repair_json(repaired_response, return_dict=True)
                    json_obj = json.dumps(json_obj, ensure_ascii=False, indent=2)
                    parsed_output = output_schema.model_validate_json(maybe_unwrap_named_root(json_obj, output_schema))
                    self.logger.info(f"Structured output JSON successfully validated...")
                except Exception: 
                    json_obj = None
                    self.logger.error(f"Structured output JSON failed validation with errors: {e}")
        return StructuredResponse(
            output_schema=output_schema,
            model_response=model_response,
            json_obj=json_obj,
            parsed_output=parsed_output,
        )

    # Tool routing + execution (unchanged from previous version)
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
        tool_descriptions = self.toolkit.get_tools_description()
        router_prompt = (
            STRICT_TOOL_ROUTING_PROMPT if strict_tool_call else TOOL_ROUTING_PROMPT
        ).format(tool_descriptions=tool_descriptions, extra_instructions=extra_instructions or "")

        routing_attempt = 0
        repair_prompt = ""
        tool_name: str = ""
        tool_inputs: Dict[str, Any] = {}
        while True:
            model_response = self.llm.ask(
                user_prompt=user_prompt if not repair_prompt else f"{user_prompt}\n\n{repair_prompt}",
                system_prompt=router_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                stream=False,
            ).choices[0].message.content or ""
            # try parsing json, in case model suggests a tool call
            obj = extract_and_repair_json(model_response, return_dict=True)
            if obj and isinstance(obj, dict) and "tool" in obj and "inputs" in obj:
                tool_name = str(obj["tool"])
                tool_inputs = obj["inputs"] if isinstance(obj["inputs"], dict) else {}
            else:
                if strict_tool_call:
                    tool_name, tool_inputs = "none", {}
                else: 
                    # return plain response if model has a choice between selecting a tool or answer directly.
                    return model_response 

            # for strict tool calling, validate tool call, and go into repair loop if invalid
            if tool_name and tool_name != "none":
                tool = self.toolkit.registry.get(tool_name)
                if tool is None:
                    repair_prompt = f"[Previous issue]\nThe tool '{tool_name}' you selected is not registered with the toolkit. Please select the right tool, double-check the name."
                    continue
                try:
                    checked = self._validate_inputs(tool_name, tool_inputs)
                except Exception as e:
                    repair_prompt = (
                        f"[Previous issue]\nFix inputs for tool '{tool_name}'. Error: {e}\nCurrent: {tool_inputs}\n"
                        "Return ONLY a JSON object for 'inputs'. Please double-check the input names and types."
                    )
                    continue
                break

            routing_attempt += 1
            if routing_attempt > self.config.retry.max_route_retries or not self.config.repair_with_llm:
                return "There was a problem selecting the right tool or repairing invalid inputs while `strict_tool_call` is enabled."

            repair_prompt = "[Previous issue]\nYour output was not valid JSON or did not select a usable tool. Return JSON only."
            time.sleep(self.config.retry.backoff_sec * routing_attempt)

        if self.verbose:
            self.logger.info(f"[ToolsCallingAgent] Selected tool: {tool_name}")
            self.logger.info(f"[ToolsCallingAgent] Raw inputs: {tool_inputs}")

        payload = {**checked, **(context or {})}
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
                self.logger.exception(f"[ToolsCallingAgent] Tool '{tool_name}' error: {e}")
            return (
                f"Error while executing tool '{tool_name}': {e}.\n"
                f"Inputs: {checked}\n"
            )

    def _validate_inputs(self, tool_name: str, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.toolkit.registry.get(tool_name)
        if tool is None: return raw_inputs or {}
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

