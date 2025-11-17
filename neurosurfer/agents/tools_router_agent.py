# neurosurfer/agents/tools_router.py

from __future__ import annotations

import json
import re
import time
import logging
import copy
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

from ..models.chat_models.base import BaseChatModel as BaseChatModel
from ..tools import Toolkit
from ..tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, TOOL_TYPE_CAST
from neurosurfer.server.schemas import ChatCompletionChunk, ChatCompletionResponse

# Config & Retry Policy
@dataclass
class RouterRetryPolicy:
    max_route_retries: int = 2        # attempts to repair/redo routing when JSON invalid or no tool
    max_tool_retries: int = 1         # attempts to rerun tool after error (with repaired inputs if possible)
    backoff_sec: float = 0.7          # linear backoff between retries

@dataclass
class ToolsRouterConfig:
    allow_input_pruning: bool = True  # drop unknown params not in ToolSpec before validation
    repair_with_llm: bool = True      # ask LLM to repair when routing/validation fails
    return_stream_by_default: bool = True
    retry: RouterRetryPolicy = field(default_factory=RouterRetryPolicy)

    # LLM defaults (can be overridden per run(...))
    temperature: float = 0.7
    max_new_tokens: int = 4000

# Tool and Input Response
class ToolAndInputsModel(BaseChatModel):
    tool_name: str
    tool_inputs: Dict[str, Any]

# Agent
class ToolsRouterAgent:
    """
    Minimal, production-ready tools router:
      1) Uses LLM to select exactly one tool + inputs (strict JSON).
      2) Validates inputs against the tool's ToolSpec (prunes unknowns if enabled).
      3) Executes the tool, proxying streaming or returning a string.
      4) Retries routing and tool execution with backoff (bounded).
    """

    def __init__(
        self,
        toolkit: Toolkit,
        llm: BaseChatModel,
        logger: logging.Logger = logging.getLogger(__name__),
        verbose: bool = False,
        specific_instructions: str = "",
        config: Optional[ToolsRouterConfig] = None,
    ):
        self.toolkit = toolkit
        self.llm = llm
        self.logger = logger
        self.verbose = verbose
        self.specific_instructions = specific_instructions
        self.config = config or ToolsRouterConfig()

    # ---------- PUBLIC API ----------
    def run(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        execute_tool: bool = True,
        strict_tool_call: bool = False,
        *,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Decide a tool with the LLM (JSON: {"tool": "...", "inputs": {...}}),
        validate/repair inputs, and execute that tool.
        - If stream=True (default from config), returns a Generator[str].
        - If stream=False, returns a str.
        Extra **kwargs are forwarded into the tool call (merged with inputs).
        """
        use_stream = self.config.return_stream_by_default if stream is None else bool(stream)
        temp = self.config.temperature if temperature is None else float(temperature)
        mnt = self.config.max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        # Routing + retries
        routing_attempt = 0
        last_error_context = ""
        tool_name: str = ""
        tool_inputs: Dict[str, Any] = {}

        while True:
            tool_name, tool_inputs = self._route(
                user_query=user_query,
                chat_history=chat_history or [],
                temperature=temp,
                max_new_tokens=mnt,
                extra_error_context=last_error_context,
                strict_tool_call=strict_tool_call
            )
            if tool_name and tool_name != "none":
                break
            routing_attempt += 1
            if routing_attempt > self.config.retry.max_route_retries or not self.config.repair_with_llm:
                # Final fallback: answer user directly with a helpful message
                return self._error_response_streaming(
                    message=f"Could not select a suitable tool for the query.\nUser query: {user_query}"
                ) if use_stream else self._error_response_text(
                    message=f"Could not select a suitable tool for the query.\nUser query: {user_query}"
                )
            # backoff and try routing again
            time.sleep(self.config.retry.backoff_sec * routing_attempt)

        if self.verbose:
            self.logger.info(f"[router] Using tool: {tool_name}")
            self.logger.info(f"[router] Raw inputs: {tool_inputs}")

        # Get tool & validate inputs with ToolSpec
        tool = self.toolkit.registry.get(tool_name)
        if tool is None:
            msg = f"Selected tool '{tool_name}' is not registered."
            return self._error_response_streaming(msg) if use_stream else self._error_response_text(msg)

        try:
            checked_inputs = self._validate_inputs(tool_name, tool_inputs)
        except Exception as e:
            # Try repair via LLM if allowed
            if self.config.repair_with_llm and routing_attempt <= self.config.retry.max_route_retries:
                last_error_context = f"Input validation error for tool '{tool_name}': {str(e)}"
                if self.verbose:
                    self.logger.warning(f"[router] {last_error_context}")
                return self.run(
                    user_query=user_query,
                    chat_history=chat_history,
                    stream=use_stream,
                    temperature=temp,
                    max_new_tokens=mnt,
                    **kwargs,
                )
            msg = f"Input validation failed for '{tool_name}': {e}"
            return self._error_response_streaming(msg) if use_stream else self._error_response_text(msg)

        # Return tool and inputs if not executing
        if not execute_tool:
            return ToolAndInputsModel(tool_name=tool_name, tool_inputs=checked_inputs)

        # Execute with bounded retries
        # payload = {"query": user_query, **kwargs, **checked_inputs}
        payload = {**checked_inputs, **kwargs}
        exec_attempt = 0
        while True:
            try:
                return self._execute_tool(tool, payload, stream=use_stream)
            except Exception as e:
                exec_attempt += 1
                if self.verbose:
                    self.logger.exception(f"[router] Tool '{getattr(tool, 'name', tool_name)}' error: {e}")
                if exec_attempt > self.config.retry.max_tool_retries or not self.config.repair_with_llm:
                    msg = f"[Tool Error] {getattr(tool, 'name', tool_name)} failed: {e}"
                    return self._error_response_streaming(msg) if use_stream else self._error_response_text(msg)
                # try again after backoff (optionally, could re-route)
                time.sleep(self.config.retry.backoff_sec * exec_attempt)

    # ---------- TOOL EXECUTION ----------
    def _execute_tool(
        self,
        tool: BaseTool,
        payload: Dict[str, Any],
        *,
        stream: bool,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Execute tool and proxy its output.
        - If the tool returns ToolResponse with a generator (stream), yield chunks.
        - If it returns text, either yield once (stream=True) or return text (stream=False).
        """
        response: ToolResponse = tool(**payload)

        # Streamed results (generator of strings or ChatCompletionChunk -> we make it strings)
        if isinstance(response.results, Generator):
            def _proxy_stream() -> Generator[str, None, None]:
                for chunk in response.results:
                    # Handle OpenAI-like ChatCompletionChunk or plain strings
                    try:
                        # ChatCompletionChunk path
                        yield getattr(chunk.choices[0].delta, "content", "")  # type: ignore[attr-defined]
                    except Exception:
                        # Plain string path
                        yield str(chunk)
            return _proxy_stream()

        # Non-stream path
        if stream:
            def _single() -> Generator[str, None, None]:
                obs = self._to_text(response.results)
                if obs:
                    yield obs
            return _single()
        else:
            return self._to_text(response.results)

    # ---------- LLM ROUTING ----------
    def _route(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]],
        *,
        temperature: float,
        max_new_tokens: int,
        extra_error_context: str = "",
        strict_tool_call: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Ask the LLM to pick a tool. Returns (tool_name, inputs_dict).
        Enforces: valid JSON with exactly {"tool": "...", "inputs": {...}}.
        """
        system_prompt = self._tools_router_system_prompt(strict_tool_call=strict_tool_call)
        routing_prompt = self._format_router_input(user_query, chat_history, extra_error_context=extra_error_context)

        resp = self.llm.ask(
            user_prompt=routing_prompt,
            system_prompt=system_prompt,
            chat_history=[],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=False,
        )
        # Align with modern response API
        try:
            raw = resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:
            # fallback for dict-shaped responses
            raw = resp["choices"][0]["message"]["content"]  # type: ignore[index]
        # if self.verbose:
        #     self.logger.info(f"[router] routing raw: {raw}")
        tool_name, inputs = self._extract_tool_json(raw)
        # if self.verbose:
        #     self.logger.info(f"[router] parsed -> tool={tool_name}, inputs={inputs}")
        return tool_name, inputs

    # ---------- PROMPTS ----------
    def _tools_router_system_prompt(self, strict_tool_call: bool = False) -> str:
        tool_descriptions = self.toolkit.get_tools_description().strip()
        return (STRICT_TOOL_ROUTING_PROMPT if strict_tool_call else TOOL_ROUTING_PROMPT).format(
            tool_descriptions=tool_descriptions,
            specific_instructions=self.specific_instructions,
        ).strip()

    def _format_router_input(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]],
        *,
        extra_error_context: str = "",
    ) -> str:
        # Keep context small and deterministic.
        hist_lines: List[str] = []
        for m in (chat_history or [])[-10:]:
            role = m.get("role", "user")
            content = m.get("content", "").replace("\n", " ").strip()
            hist_lines.append(f"{role}: {content}")
        history_block = "\n".join(hist_lines)
        if extra_error_context:
            error_block = f"\n\nHint (previous error): {extra_error_context}\n"
        else:
            error_block = ""
        return f"User message:\n{user_query}\n\nRecent chat (newest last):\n{history_block}{error_block}"

    # ---------- INPUT VALIDATION ----------

    def _validate_inputs(self, tool_name: str, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean inputs using ToolSpec. Optionally prune unknown keys.
        """
        spec: Optional[ToolSpec] = getattr(self.toolkit, "specs", {}).get(tool_name)
        if spec is None:
            # No spec available; pass through
            return raw_inputs or {}

        inputs = dict(raw_inputs or {})
        if self.config.allow_input_pruning:
            allowed = {p.name for p in spec.inputs}
            inputs = {k: v for k, v in inputs.items() if k in allowed}

        # Raises on type/required/extras (when pruning disabled)
        checked = spec.check_inputs(inputs)
        return checked

    # ---------- PARSING UTILS ----------

    def _extract_tool_json(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse {"tool":"...", "inputs":{...}} from arbitrary LLM output.
        """
        # strip code fences & surrounding cruft
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

        # parse the inputs to correct types
        def _parse_inputs(obj: dict) -> Dict[str, Any]:
            inputs: dict = copy.deepcopy(obj["inputs"])
            for param, value in obj["inputs"].items():
                tool: Optional[BaseTool] = self.toolkit.registry.get(obj["tool"], None)
                if tool is None: continue
                for p in tool.spec.inputs:
                    if p.name == param:
                        inputs[param] = TOOL_TYPE_CAST[p.type](value)
            return inputs

        # fast path
        obj: dict = self._try_json(cleaned)
        if self._looks_like_decision(obj):
            # parse the inputs to correct types
            inputs: dict = _parse_inputs(obj)
            return obj["tool"], inputs

        # scan for first JSON object
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            obj: dict = self._try_json(match.group(0))
            if self._looks_like_decision(obj):
                return obj["tool"], _parse_inputs(obj)
        return "none", {}

    @staticmethod
    def _looks_like_decision(obj: Any) -> bool:
        return isinstance(obj, dict) and "tool" in obj and "inputs" in obj and isinstance(obj["inputs"], dict)

    @staticmethod
    def _try_json(s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            return None

    # ---------- ERROR FALLBACKS ----------

    def _error_response_text(self, message: str) -> str:
        system_prompt = (
            "Provide a concise, helpful response to the user. "
            "Do not mention internal errors. If needed, ask the user to rephrase."
        )
        try:
            resp = self.llm.ask(
                system_prompt=system_prompt,
                user_prompt=f"{message}",
                chat_history=[],
                temperature=0.5,
                max_new_tokens=512,
                stream=False,
            )
            return resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:
            # Last resort: return the message
            return message

    def _error_response_streaming(self, message: str) -> Generator[str, None, None]:
        system_prompt = (
            "Provide a concise, helpful response to the user. "
            "Do not mention internal errors. If needed, ask the user to rephrase."
        )
        try:
            gen = self.llm.ask(
                system_prompt=system_prompt,
                user_prompt=f"{message}",
                chat_history=[],
                temperature=0.5,
                max_new_tokens=512,
                stream=True,
            )
            for chunk in gen:
                try:
                    yield getattr(chunk.choices[0].delta, "content", "")  # type: ignore[attr-defined]
                except Exception:
                    yield str(chunk)
        except Exception:
            yield message

    # ---------- MISC UTILS ----------

    @staticmethod
    def _to_text(x: Union[str, ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]) -> str:
        if isinstance(x, str):
            return x
        # ChatCompletionResponse (non-stream)
        try:
            return x.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    @staticmethod
    def _ensure_generator(x: Any) -> Generator[str, None, None]:
        if x is None:
            if False:
                yield ""  # pragma: no cover
            return
        if isinstance(x, str):
            def _g():
                yield x
            return _g()
        if isinstance(x, Iterable):
            def _g():
                for item in x:
                    yield str(item)
            return _g()
        def _g():
            yield str(x)
        return _g()


TOOL_ROUTING_PROMPT = """You are a stateless tool router. 
You may either call a tool or respond directly with a natural language message — whichever best fits the user query.

If you decide to call a tool, respond **only** with one-line valid JSON in the exact format below:
{{"tool": "<tool_name>", "inputs": {{<param>: <value>}}}}

If you decide to respond directly, emit your message as a plain string (not JSON).

Rules:
- Choose at most ONE tool per request.
- Use only explicit parameters defined by that tool. Do NOT invent or rename parameters.
- Include only required parameters unless an optional one is clearly implied.
- If no tool fits the request or inputs are ambiguous, output:
  {{"tool": "none", "inputs": {{}}}}
- Otherwise, respond in plain text when a natural language answer is more suitable.

TOOL CATALOG:
{tool_descriptions}

{specific_instructions}
"""


STRICT_TOOL_ROUTING_PROMPT = """You are a stateless tool router.
Your task is to select exactly ONE tool from the catalog below and output STRICT JSON describing how to call it.

Always respond with a single one-line valid JSON object:
{{"tool": "<tool_name>", "inputs": {{<param>: <value>}}}}

Rules:
- Output MUST contain exactly the keys "tool" and "inputs".
- Select at most one tool; if none applies or inputs are unclear, use:
  {{"tool": "none", "inputs": {{}}}}
- Use only parameters explicitly defined by that tool — do NOT invent, rename, or add extra fields.
- Include only required parameters unless an optional one is obviously needed.
- Do NOT produce natural language; emit JSON only.

TOOL CATALOG:
{tool_descriptions}

{specific_instructions}
"""