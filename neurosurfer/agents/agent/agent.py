from __future__ import annotations

import json, time, logging
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union, Type

from pydantic import BaseModel as PydModel

from neurosurfer.models.chat_models.base import BaseModel as ChatBaseModel
from neurosurfer.server.schemas import ChatCompletionChunk, ChatCompletionResponse
from neurosurfer.tools import Toolkit
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, TOOL_TYPE_CAST

from .config import ToolsCallingAgentConfig
from .router_prompts import TOOL_ROUTER_PROMPT, STRICT_TOOL_ROUTER_PROMPT
from .schema_utils import (
    pydantic_model_from_outputs,
    structure_block,
    json_first_object,
    apply_synonyms,
)

class ToolsCallingAgent:
    """
    A reusable agent that:
      - Works as plain LLM (un/streamed)
      - Or as a tool-calling router when `tools` are provided
      - Optionally produces structured output (typed outputs list)
      - Validates tool inputs via ToolSpec; supports synonyms + LLM repair
      - Returns a consistent envelope in tool mode
    """

    def __init__(
        self,
        llm: ChatBaseModel,
        toolkit: Optional[Toolkit] = None,
        *,
        config: Optional[ToolsCallingAgentConfig] = None,
        logger: logging.Logger = logging.getLogger(__name__),
        verbose: bool = False,
    ):
        self.llm = llm
        self.toolkit = toolkit or Toolkit()
        self.config = config or ToolsCallingAgentConfig()
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
        query: Optional[str] = None,     # alias
        tools: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,  # e.g. ["num1: float", "op: enum{add,subtract}"]
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        _route_extra_instructions: str = "",
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        use_stream = self.config.return_stream_by_default if stream is None else bool(stream)
        temp = float(self.config.temperature if temperature is None else temperature)
        mnt = int(self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        sys_prompt = system_prompt or "You are a helpful assistant. Be concise and precise."
        usr_prompt = (user_prompt if user_prompt is not None else query) or ""

        # Tool mode?
        tool_names = tools or []
        if tool_names:
            return self._route_and_call(
                user_prompt=usr_prompt,
                tool_names=tool_names,
                extra_instructions=_route_extra_instructions,
                temperature=temp,
                max_new_tokens=mnt,
                use_stream=use_stream,
                context=context or {},
            )

        # Plain LLM mode (structured or free text)
        if outputs and self._outputs_is_structured_list(outputs):
            Model = pydantic_model_from_outputs(outputs, model_name="AgentStructuredOutput")
            block = structure_block(outputs, title=Model.__name__)
            contract = (
                "## Structured Output Contract\n"
                "- Return a single JSON object matching the structure below.\n"
                "- Valid JSON only (RFC 8259). No code fences, no markdown, no explanations.\n"
                f'- Do NOT wrap the object under a named key like "{Model.__name__}". Return the object itself.\n\n'
                "### Structure (read-only guidance)\n"
                f"{block}\n"
            )
            sys = (sys_prompt + "\n\n" + contract).strip()
            parsed = self.llm.ask(
                user_prompt=usr_prompt,
                system_prompt=sys,
                temperature=temp,
                max_new_tokens=mnt,
                stream=False,  # force non-stream for structured
                output_schema=Model,
                strict_json=self.config.strict_json,
                on_parse_error=self.config.on_parse_error,
                max_repair_attempts=self.config.max_repair_attempts,
            )
            return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed

        # Free-text mode
        resp = self.llm.ask(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            temperature=temp,
            max_new_tokens=mnt,
            stream=use_stream,
        )
        if use_stream:
            def _gen() -> Generator[str, None, None]:
                for chunk in resp:
                    yield getattr(chunk.choices[0].delta, "content", "")
            return _gen()
        else:
            return {"text": resp.choices[0].message.content}

    # ---------------------------------------------------------------------
    # Internals: tool routing + execution + validation
    # ---------------------------------------------------------------------

    def _route_and_call(
        self,
        *,
        user_prompt: str,
        tool_names: List[str],
        extra_instructions: str,
        temperature: float,
        max_new_tokens: int,
        use_stream: bool,
        context: Dict[str, Any],
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        catalog = self._catalog(tool_names)
        router_prompt = (
            STRICT_TOOL_ROUTER_PROMPT if self.config.strict_tool_call else TOOL_ROUTER_PROMPT
        ).format(catalog=catalog, extra=extra_instructions or "")

        routing_attempt = 0
        last_err = ""
        tool_name: str = ""
        tool_inputs: Dict[str, Any] = {}

        # -------- Routing (bounded retries) --------
        while True:
            resp = self.llm.ask(
                user_prompt=user_prompt if not last_err else f"{user_prompt}\n\n[Previous issue]\n{last_err}",
                system_prompt=router_prompt,
                temperature=temperature,
                max_new_tokens=min(256, max_new_tokens),
                stream=False,
            )
            raw = (resp.choices[0].message.content or "").strip()

            obj = json_first_object(raw)
            if obj and isinstance(obj, dict) and "tool" in obj and "inputs" in obj:
                tool_name = str(obj["tool"])
                tool_inputs = obj["inputs"] if isinstance(obj["inputs"], dict) else {}
                tool_inputs = apply_synonyms(tool_inputs, self.config.synonyms)
            else:
                if self.config.strict_tool_call:
                    tool_name, tool_inputs = "none", {}
                else:
                    # relaxed: if router replies in text, return it
                    if not tool_names:
                        return {"text": raw}
                    tool_name, tool_inputs = "none", {}

            if tool_name and tool_name != "none":
                break

            routing_attempt += 1
            if routing_attempt > self.config.retry.max_route_retries or not self.config.repair_with_llm:
                if raw and not obj:
                    return {"selected_tool": "none", "inputs": {}, "text": raw, "returns": None, "final": False, "extras": {}}
                return {"selected_tool": "none", "inputs": {}, "text": "No suitable tool selected.", "returns": None, "final": False, "extras": {}}

            last_err = "Your output was not valid JSON or did not select a usable tool. Return JSON only."
            time.sleep(self.config.retry.backoff_sec * routing_attempt)

        if self.verbose:
            self.logger.info(f"[ToolsCallingAgent] Selected tool: {tool_name}")
            self.logger.info(f"[ToolsCallingAgent] Raw inputs: {tool_inputs}")

        # -------- Validate inputs via ToolSpec (with one repair cycle) --------
        tool = self.toolkit.registry.get(tool_name)
        if tool is None:
            return {"selected_tool": tool_name, "inputs": tool_inputs, "text": f"Tool '{tool_name}' not registered.", "returns": None, "final": False, "extras": {}}

        try:
            checked = self._validate_inputs(tool_name, tool_inputs)
        except Exception as e:
            if not self.config.repair_with_llm:
                return {"selected_tool": tool_name, "inputs": tool_inputs, "text": f"Input validation failed: {e}", "returns": None, "final": False, "extras": {}}

            fix_resp = self.llm.ask(
                user_prompt=f"Fix inputs for tool '{tool_name}'. Error: {e}\nCurrent: {tool_inputs}\nReturn ONLY a JSON object for 'inputs'.",
                system_prompt="Return only valid JSON with corrected inputs. No commentary.",
                temperature=0.0,
                max_new_tokens=128,
                stream=False,
            )
            fixed = json_first_object(fix_resp.choices[0].message.content or "{}") or {}
            if isinstance(fixed, dict) and "inputs" in fixed and isinstance(fixed["inputs"], dict):
                tool_inputs = apply_synonyms(fixed["inputs"], self.config.synonyms)
                checked = self._validate_inputs(tool_name, tool_inputs)
            else:
                return {"selected_tool": tool_name, "inputs": tool_inputs, "text": f"Could not repair tool inputs.", "returns": None, "final": False, "extras": {}}

        # -------- Execute tool (with retries) --------
        payload = {**checked, **(context or {})}
        attempt = 0
        while True:
            try:
                resp = tool(**payload)
                extras = resp.extras or {}

                # streaming?
                if hasattr(resp.observation, "__iter__") and not isinstance(resp.observation, (str, bytes)):
                    def _stream() -> Generator[str, None, None]:
                        for chunk in resp.observation:
                            try:
                                yield getattr(chunk.choices[0].delta, "content", "")
                            except Exception:
                                yield str(chunk)
                    return _stream()

                # non-stream: normalize text + try parse JSON
                text = self._to_text(resp.observation)
                parsed_json: Optional[Dict[str, Any]] = None
                try:
                    pj = json.loads(text)
                    if isinstance(pj, dict):
                        parsed_json = pj
                except Exception:
                    pass

                return {
                    "selected_tool": tool_name,
                    "inputs": checked,
                    "text": text,
                    "returns": parsed_json,
                    "final": bool(resp.final_answer),
                    "extras": extras,
                }

            except Exception as e:
                attempt += 1
                if self.verbose:
                    self.logger.exception(f"[ToolsCallingAgent] Tool '{tool_name}' error: {e}")
                if attempt > self.config.retry.max_tool_retries or not self.config.repair_with_llm:
                    return {
                        "selected_tool": tool_name,
                        "inputs": checked,
                        "text": f"[Tool Error] {tool_name} failed: {e}",
                        "returns": None,
                        "final": False,
                        "extras": {},
                    }
                time.sleep(self.config.retry.backoff_sec * attempt)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _catalog(self, tool_names: List[str]) -> str:
        blocks = []
        for name in tool_names:
            t = self.toolkit.registry.get(name)
            if not t:
                continue
            blocks.append(t.get_tool_description().strip())
        return "\n\n".join(blocks)

    def _validate_inputs(self, tool_name: str, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        spec: Optional[ToolSpec] = getattr(self.toolkit, "specs", {}).get(tool_name)
        if spec is None:
            return raw_inputs or {}

        inputs = dict(raw_inputs or {})
        if self.config.allow_input_pruning:
            allowed = {p.name for p in spec.inputs}
            inputs = {k: v for k, v in inputs.items() if k in allowed}

        # Cast by ToolParam types (string->float, etc.) for robustness:
        for p in spec.inputs:
            if p.name in inputs and p.type in TOOL_TYPE_CAST:
                try:
                    inputs[p.name] = TOOL_TYPE_CAST[p.type](inputs[p.name])
                except Exception:
                    pass  # spec.check_inputs will raise if incompatible

        checked = spec.check_inputs(inputs)
        return checked

    @staticmethod
    def _outputs_is_structured_list(outputs: Any) -> bool:
        if not isinstance(outputs, list):
            return False
        if len(outputs) == 1 and outputs[0] == "text":
            return False
        return all(isinstance(x, str) and (":" in x or x.strip() == "text") for x in outputs)

    @staticmethod
    def _to_text(x: Union[str, ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]) -> str:
        if isinstance(x, str):
            return x
        try:
            return x.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
