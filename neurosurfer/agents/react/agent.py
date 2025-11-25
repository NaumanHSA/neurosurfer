import logging, traceback, json, re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Generator, Optional
from types import GeneratorType

from neurosurfer.tracing import Tracer, TracerConfig, TraceStepContext
from neurosurfer.server.schemas import ChatCompletionChunk, ChatCompletionResponse
from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools import Toolkit
from neurosurfer.tools.base_tool import ToolResponse
from neurosurfer.config import config

from ..common.utils import normalize_response, rprint
from .base import BaseAgent
from .parser import ToolCallParser
from .types import ToolCall, ReactAgentResponse
from .exceptions import ToolCallParseError, ToolExecutionError
from .retry import RetryPolicy
from .history import History
from .memory import EphemeralMemory
from .scratchpad import REACT_AGENT_PROMPT, REPAIR_ACTION_PROMPT
from .config import ReActConfig


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
        id: str = "ReAct_Agent",
        llm: BaseChatModel = None,
        toolkit: Optional[Toolkit] = None,
        *,
        specific_instructions: str = "",
        config: Optional[ReActConfig] = None,
        logger: logging.Logger = logging.getLogger(__name__),
        verbose: bool = False,
        tracer: Optional[Tracer] = None,
        log_traces: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.id = id
        self.llm = llm
        self.toolkit = toolkit
        self.specific_instructions = specific_instructions
        self.config = config or ReActConfig()
        self.logger = logger
        self.verbose = verbose
        self.log_traces = log_traces
        if self.llm is None:
            raise ValueError("llm must be provided")

         # Tracing setup
        # Base tracer that actually records and log steps (RichTracer by default).
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "generic_agent",
                "agent_config": self.config,
                "model": self.llm.model_name,
                "toolkit": toolkit is not None,
                "verbose": verbose,
                "log_steps": self.log_traces,
            },
            logger_=logger,
        )
        self.parser = ToolCallParser()
        self.memory = EphemeralMemory()
        self.raw_results = ""
        self.schema_context = ""  # keep if you want to inject schemas
        self._last_error: Optional[str] = None

    # ---------- Public API ----------
    def run(
        self,
        *,
        query: Optional[str] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        specific_instructions: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        _route_extra_instructions: str = "",
        reset_tracer: bool = True,
    ) -> ReactAgentResponse:
        """
        Run the ReAct agent.

        - If stream=True: returns ReactAgentResponse with a Generator[str]
          that yields tokens as they are produced.
        - If stream=False: returns ReactAgentResponse with the full string
          (we still use streaming under the hood, but we buffer it).
        """
        if reset_tracer:
            self.tracer.reset()

        # Resolve config defaults
        stream = self.config.return_stream_by_default if stream is None else bool(stream)
        temperature = float(self.config.temperature if temperature is None else temperature)
        max_new_tokens = int(self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        specific_instructions = (
            self.specific_instructions
            if specific_instructions is None
            else specific_instructions or ""
        )
        system_prompt = self._system_prompt(specific_instructions)

        rprint("🧠 Thinking...", color="yellow")
        if self.log_traces:
            rprint(f"\n\\[{self.id}] Tracing Start!")

        with self.tracer(
            agent_id=self.id,
            kind="react_agent",
            label="react_agent.run",
            inputs={
                "agent_type": type(self).__name__,
                "has_toolkit": bool(self.toolkit),
                "stream": stream,
                "specific_instructions": specific_instructions,
                "scratchpad": system_prompt,
            },
        ):
            # -------- Streaming path --------
            if stream:
                final_response = self._run_loop(
                    query=query,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    system_prompt=system_prompt,
                )
            # -------- Non-streaming path --------
            else:
                final_response = ""
                for chunk in self._run_loop(
                    query=query,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    system_prompt=system_prompt,
                ):
                    final_response += chunk

            if self.log_traces:
                rprint(f"\\[{self.id}] Tracing End!\n")

            # Print final response (like your normal Agent)
            if isinstance(final_response, str):
                rprint("Final response:", color="green")
                rprint(final_response.strip(), rich=False)
            return ReactAgentResponse(response=final_response, traces=self.tracer.results)

    def stop_generation(self):
        self.logger.info("[ReActAgent] Stopping generation...")
        try:
            self.llm.stop_generation()
        finally:
            super().stop_generation()

    def update_toolkit(self, toolkit: Toolkit) -> None:
        self.toolkit = toolkit

    # ---------- Core loop ----------
    def _run_loop(
        self,
        query: str, 
        temperature: float, 
        max_new_tokens: int,
        system_prompt: str,
    ) -> Generator[str, None, str]:

        history = History()
        final_answer = ""
        self.stop_event = False
        iteration = 0
        while not self.stop_event:
            reasoning_prompt = self._build_prompt(query, history)
            reason_tracer = self.tracer(
                agent_id=self.id,
                kind="llm.call",
                label="react_agent.loop.reason_llm_call",
                inputs={
                    "iteration": iteration,                          # ReAct loop iteration
                    "query": query,
                    "query_len": len(query),
                    "history_len": len(history.as_text()),
                    "reasoning_prompt": reasoning_prompt,
                    "reasoning_prompt_len": len(reasoning_prompt),
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                },
            ).start()

            streaming_response = self.llm.ask(
                user_prompt=reasoning_prompt,
                system_prompt=system_prompt,
                chat_history=[],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                stream=True
            )

            response, final_started = "", False
            for chunk in streaming_response:
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
                reason_tracer.outputs(final_answer=final_answer)
                if not self.config.skip_special_tokens:
                    yield self.delims.eof
                break
            
            # No final answer yet, try to parse an Action
            tool_call = self._decide_tool_call(response, query, history)
            if tool_call is None or tool_call.tool is None:
                # agent believes no tool is required and no final answer was streamed; ask LLM to produce final
                history.append(response)
                # Add a gentle nudge: produce final answer now
                final = self._force_final_answer(query, history, temperature, max_new_tokens)
                yield self.delims.sof + final + self.delims.eof
                final_answer = final
                break
            
            reason_tracer.log(message=f"[🔧] Tool Selected: {tool_call.tool}", type="info")

            tool_tracer = self.tracer(
                agent_id=self.id,
                kind="llm.call",
                label="react_agent.loop.tool_execute",
                inputs={
                    "tool_name": tool_call.tool,
                    "tool_inputs": tool_call.inputs
                },
            ).start()

            history.append(response)
            # Execute tool safely with bounded retries
            # core.py, inside _run_loop, replacing the "execute tool" part
            tool_response = self._try_execute_tool(tool_call, tool_tracer)
            tool_results = normalize_response(tool_response.results)

            if isinstance(tool_results, GeneratorType):
                # live stream to the user and accumulate
                results_text = ""
                if tool_response.final_answer and not self.config.skip_special_tokens:
                    yield self.delims.sof
                for chunk in tool_results:
                    results_text += chunk
                    yield chunk
                if tool_response.final_answer:
                    if not self.config.skip_special_tokens:
                        yield self.delims.eof
                    final_answer = results_text
                    break
                # not final
                history.append(f"results: {results_text}")
                if self.config.verbose:
                    rprint(f"[bold]results:[/bold] {results_text}")
            else:
                # plain string
                results_text = tool_results
                if tool_response.final_answer:
                    final_answer = results_text
                    if not self.config.skip_special_tokens:
                        final_answer = self.delims.sof + results_text + self.delims.eof
                    yield final_answer
                    break
                history.append(f"results: {results_text}")
                if self.config.verbose:
                    rprint(f"[bold]results:[/bold] {results_text}")

            tool_tracer.log(f"\n[🔧] Tool Result: {results_text[:150]}...", type="info")
            tool_tracer.close()
            iteration += 1

        tool_tracer.log(f"\n[🔧] Final Tool Result: {final_answer[:150]}...", type="info")
        tool_tracer.close()
        reason_tracer.close()
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
                spec.check_inputs(sanitized, relax=True)
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
    def _try_execute_tool(self, tool_call: ToolCall, tool_tracer: TraceStepContext) -> ToolResponse:
        tool_name = tool_call.tool
        tool = self.toolkit.registry[tool_name]

        attempts = 0
        last_err = None
        while attempts <= self.config.retry.max_tool_errors:
            try:
                tool_tracer.log(f"\n[🔧] Executing Tool: {tool_name}, Attempt: {attempts}, [📤] Inputs: {tool_call.inputs.keys()}", type="info")
                all_inputs = {**tool_call.inputs, **self.memory.items()}
                tool_response: ToolResponse = tool(**all_inputs)
                self.memory.clear()
                # write extras to memory for next step
                for k, v in tool_response.extras.items():
                    self.memory.set(k, v)
                return tool_response

            except Exception as e:
                last_err = str(e)
                tool_tracer.log(f"Error Executing Tool: {tool_name} -> {last_err}", type="error")
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
        tool_tracer.log(f"Tool Execution Failed: {tool_name}, Reason: Retries Exceeded", type="warning")
        return ToolResponse(
            results=f"[tool:{tool_name}] failed after retries: {last_err}",
            final_answer=False,
            extras={}
        )

    def _mk_error_history(self, tool_name: str, tool_call: ToolCall, error: str) -> History:
        h = History()
        h.append(f"Thought: Previous tool call to {tool_name} failed.")
        h.append(f"Action: {json.dumps({'tool': tool_name, 'inputs': tool_call.inputs, 'final_answer': tool_call.final_answer})}")
        h.append(f"results: ERROR -> {error}")
        return h

    # ---------- Prompts ----------
    def _system_prompt(self, specific_instructions: str) -> str:
        from .scratchpad import REACT_AGENT_PROMPT
        tool_desc = self.toolkit.get_tools_description().strip()
        return REACT_AGENT_PROMPT.format(
            tool_descriptions=tool_desc,
            specific_instructions=specific_instructions
        )

    def _build_prompt(self, user_query: str, history: History) -> str:
        prompt = f"# User Query:\n{user_query}\n"
        prompt += history.to_prompt()
        prompt += "\n# Next Steps:\nWhat should you do next?\n" \
                  "If you think the answer is ready, generate a complete Final Answer independent of the history.\n"
        return prompt
