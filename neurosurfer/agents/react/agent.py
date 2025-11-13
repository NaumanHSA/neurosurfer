import logging, traceback, json, re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Generator, Optional
from types import GeneratorType

from rich import print as rprint

from neurosurfer.server.schemas import ChatCompletionChunk, ChatCompletionResponse
from neurosurfer.models.chat_models.base import BaseModel
from neurosurfer.tools import Toolkit
from neurosurfer.tools.base_tool import ToolResponse
from neurosurfer.config import config

from .base import BaseAgent
from .parser import ToolCallParser
from .types import ToolCall
from .exceptions import ToolCallParseError, ToolExecutionError
from .retry import RetryPolicy
from .history import History
from .memory import EphemeralMemory
from ..common.utils import normalize_tool_observation
from .scratchpad import REACT_AGENT_PROMPT, REPAIR_ACTION_PROMPT


@dataclass
class ReActConfig:
    temperature: float = config.base_model.temperature
    max_new_tokens: int = config.base_model.max_new_tokens
    verbose: bool = True
    allow_input_pruning: bool = True     # drop extra inputs not in ToolSpec
    repair_with_llm: bool = True         # ask LLM to repair invalid Action
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    skip_special_tokens: bool = False


class ReActAgent(BaseAgent):
    """
    Production-ready ReAct Agent with:
    - tolerant Action parsing
    - input sanitization vs ToolSpec (drop extras or repair)
    - bounded retries on parse & tool errors
    - reusable base class & utilities
    """
    def __init__(
        self,
        toolkit: Toolkit,
        llm: BaseModel,
        logger: logging.Logger = logging.getLogger(__name__),
        specific_instructions: str = "",
        config: Optional[ReActConfig] = None
    ) -> None:
        super().__init__()
        self.toolkit = toolkit
        self.llm = llm
        self.logger = logger
        self.specific_instructions = specific_instructions
        self.config = config or ReActConfig()
        self.parser = ToolCallParser()
        self.memory = EphemeralMemory()
        self.raw_results = ""
        self.schema_context = ""  # keep if you want to inject schemas
        self._last_error: Optional[str] = None

    # ---------- Public API ----------
    def run(self, user_query: str, **kwargs: Any) -> Generator[str, None, str]:
        # kwargs override config temporarily
        temperature = kwargs.get("temperature", self.config.temperature)
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)

        return self._run_loop(
            user_query=user_query,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def stop_generation(self):
        self.logger.info("[ReActAgent] Stopping generation...")
        try:
            self.llm.stop_generation()
        finally:
            super().stop_generation()

    def update_toolkit(self, toolkit: Toolkit) -> None:
        self.toolkit = toolkit

    # ---------- Core loop ----------
    def _run_loop(self, user_query: str, temperature: float, max_new_tokens: int) -> Generator[str, None, str]:
        history = History()
        final_answer = ""
        self.stop_event = False

        system_prompt = self._system_prompt()

        while not self.stop_event:
            reasoning_prompt = self._build_prompt(user_query, history)
            yield "\n\n[🧠] Chain of Thoughts...\n"
            stream = self.llm.ask(
                user_prompt=reasoning_prompt,
                system_prompt=system_prompt,
                chat_history=[],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                stream=True
            )

            response, final_started = "", False
            for chunk in stream:
                if chunk.choices[0].finish_reason == "stop":
                    break
                part = chunk.choices[0].delta.content
                response += part

                # stream final answer if the marker appears
                if not final_started and self.delims.sof in response:
                    final_started = True
                    prefix, suffix = response.split(self.delims.sof, 1)
                    if suffix.strip():
                        to_emit = self.delims.sof + suffix
                        if self.config.skip_special_tokens:
                            to_emit = to_emit.strip(self.delims.sof)
                        yield to_emit
                        final_answer += to_emit
                elif final_started:
                    if self.delims.eof in part:
                        before, _after = part.split(self.delims.eof, 1)
                        to_emit = self.delims.eof + before
                        if self.config.skip_special_tokens:
                            to_emit = to_emit.strip(self.delims.eof)
                        yield to_emit
                        final_answer += to_emit
                    else:
                        yield part
                        final_answer += part
                else:
                    yield part

            if final_started: 
                if not self.config.skip_special_tokens:
                    yield self.delims.eof
                break
            
            # No final answer yet, try to parse an Action
            tool_call = self._decide_tool_call(response, user_query, history)
            if tool_call is None or tool_call.tool is None:
                # agent believes no tool is required and no final answer was streamed; ask LLM to produce final
                history.append(response)
                # Add a gentle nudge: produce final answer now
                final = self._force_final_answer(user_query, history, temperature, max_new_tokens)
                yield self.delims.sof + final + self.delims.eof
                final_answer = final
                break

            history.append(response)
            # Execute tool safely with bounded retries
            # core.py, inside _run_loop, replacing the "execute tool" part
            tool_response = self._try_execute_tool(tool_call)
            obs = normalize_tool_observation(tool_response.observation)

            if isinstance(obs, GeneratorType):
                # live stream to the user and accumulate
                observation_text = ""
                if tool_response.final_answer:
                    if not self.config.skip_special_tokens:
                        yield self.delims.sof
                for chunk in obs:
                    observation_text += chunk
                    yield chunk
                if tool_response.final_answer:
                    if not self.config.skip_special_tokens:
                        yield self.delims.eof
                    final_answer = observation_text
                    break
                # not final
                history.append(f"Observation: {observation_text}")
                if self.config.verbose:
                    rprint(f"[bold]Observation:[/bold] {observation_text}")
            else:
                # plain string
                observation_text = obs
                if tool_response.final_answer:
                    final_answer = observation_text
                    if not self.config.skip_special_tokens:
                        final_answer = self.delims.sof + observation_text + self.delims.eof
                    yield final_answer
                    break
                history.append(f"Observation: {observation_text}")
                if self.config.verbose:
                    rprint(f"[bold]Observation:[/bold] {observation_text}")

        # self.logger.info(f"[ReActAgent] Stopped -> Final answer length: {len(final_answer)}")
        return final_answer or "I couldn't determine the answer."

    # ---------- Decision & Repair ----------
    def _decide_tool_call(self, response: str, user_query: str, history: History) -> Optional[ToolCall]:
        # 1) parse tolerant
        try:
            tc = self.parser.extract(response)
        except ToolCallParseError as e:
            self._last_error = str(e)
            if self.config.repair_with_llm:
                return self._repair_action(user_query, history, error_message=str(e))
            return None

        if tc is None:
            # no Action block; give the model one chance to repair if enabled
            if self.config.repair_with_llm:
                return self._repair_action(user_query, history, error_message="No Action block found.")
            return None

        # 2) sanitize inputs vs ToolSpec (drop extras if allowed, or repair)
        if tc.tool is not None:
            sanitized, dropped, err = self._sanitize_inputs(tc.tool, tc.inputs)
            if err:
                # Missing required or bad types -> try repair
                if self.config.repair_with_llm:
                    return self._repair_action(user_query, history, error_message=str(err))
                return None
            if dropped and self.config.verbose:
                rprint(f"[yellow][agent] Dropped extra inputs for '{tc.tool}': {sorted(dropped)}[/yellow]")
            tc.inputs = sanitized
        return tc

    def _sanitize_inputs(self, tool_name: str, inputs: Dict[str, Any]):
        tool = self.toolkit.registry.get(tool_name)
        if not tool:
            return inputs, set(), ValueError(f"Unknown tool '{tool_name}'")

        spec = self.toolkit.specs[tool_name]
        allowed_names = {p.name for p in spec.inputs}  # ToolParam list
        extras = set(inputs.keys()) - allowed_names
        sanitized = dict(inputs)

        if extras and self.config.allow_input_pruning:
            # silently drop extras (like your 'fix' flag for lint)
            for k in extras:
                sanitized.pop(k, None)
            # now validate using spec (required, types)
            try:
                spec.check_inputs(sanitized)
            except Exception as e:
                return sanitized, extras, e
            return sanitized, extras, None

        # strict path: validate directly so the error bubbles with names
        try:
            spec.check_inputs(inputs)
            return inputs, set(), None
        except Exception as e:
            return inputs, extras, e

    def _repair_action(self, user_query: str, history: History, error_message: str) -> Optional[ToolCall]:
        tool_desc = self.toolkit.get_tools_description().strip()
        prompt = REPAIR_ACTION_PROMPT.format(
            user_query=user_query,
            history=history.as_text(),
            tool_descriptions=tool_desc,
            error_message=error_message
        )
        resp = self.llm.ask(
            user_prompt=prompt,
            system_prompt="You repair invalid tool calls. Output only the Action JSON line.",
            chat_history=[],
            temperature=0.2,
            max_new_tokens=300,
            stream=False
        )
        text = resp.choices[0].message.content
        # Try to parse repaired action
        try:
            repaired = self.parser.extract(text)
        except Exception:
            return None
        return repaired

    def _force_final_answer(self, user_query: str, history: History, temperature: float, max_new_tokens: int) -> str:
        """If no Action and no final streamed, ask for a direct final answer."""
        prompt = (
            f"# User Query:\n{user_query}\n"
            f"{history.to_prompt()}"
            "\n# Next Steps:\nProduce a complete final answer now in one message."
        )
        resp = self.llm.ask(
            user_prompt=prompt,
            system_prompt="You finalize answers succinctly and helpfully.",
            chat_history=[],
            temperature=max(0.2, temperature - 0.3),
            max_new_tokens=min(max_new_tokens, 1200),
            stream=False
        )
        return resp.choices[0].message.content

    # ---------- Tool Exec with retries ----------
    def _try_execute_tool(self, tool_call: ToolCall) -> ToolResponse:
        tool_name = tool_call.tool
        tool = self.toolkit.registry[tool_name]

        attempts = 0
        last_err = None
        while attempts <= self.config.retry.max_tool_errors:
            try:
                if self.config.verbose:
                    rprint(f"\n[🔧] Tool: {tool_name}\n[📤] Inputs: {tool_call.inputs}")

                all_inputs = {**tool_call.inputs, **self.memory.items()}
                tool_response = tool(**all_inputs)
                self.memory.clear()
                # write extras to memory for next step
                for k, v in tool_response.extras.items():
                    self.memory.set(k, v)
                return tool_response

            except Exception as e:
                last_err = str(e)
                self.logger.error(f"[Tool:{tool_name}] error: {last_err}")
                if attempts >= self.config.retry.max_tool_errors:
                    break
                if self.config.repair_with_llm:
                    repaired = self._repair_action(
                        user_query=f"Tool failure for {tool_name}",
                        history=self._mk_error_history(tool_name, tool_call, last_err),
                        error_message=last_err
                    )
                    if repaired and repaired.tool:
                        tool_call = repaired
                attempts += 1
                self.config.retry.sleep(attempts)
        # Synthesize a failure ToolResponse (non-final)
        return ToolResponse(
            observation=f"[tool:{tool_name}] failed after retries: {last_err}",
            final_answer=False,
            extras={}
        )

    def _mk_error_history(self, tool_name: str, tool_call: ToolCall, error: str) -> History:
        h = History()
        h.append(f"Thought: Previous tool call to {tool_name} failed.")
        h.append(f"Action: {json.dumps({'tool': tool_name, 'inputs': tool_call.inputs, 'final_answer': tool_call.final_answer})}")
        h.append(f"Observation: ERROR -> {error}")
        return h

    # ---------- Prompts ----------
    def _system_prompt(self) -> str:
        from .scratchpad import REACT_AGENT_PROMPT
        tool_desc = self.toolkit.get_tools_description().strip()
        return REACT_AGENT_PROMPT.format(
            tool_descriptions=tool_desc,
            specific_instructions=self.specific_instructions
        )

    def _build_prompt(self, user_query: str, history: History) -> str:
        prompt = f"# User Query:\n{user_query}\n"
        prompt += history.to_prompt()
        prompt += "\n# Next Steps:\nWhat should you do next?\n" \
                  "If you think the answer is ready, generate a complete Final Answer independent of the history.\n"
        return prompt
