import logging, traceback, json, re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Generator, Optional
from types import GeneratorType

from neurosurfer.tracing import Tracer, TracerConfig, TraceStepContext
from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools import Toolkit
from neurosurfer.tools.base_tool import ToolResponse

from ..common import AgentMemory
from ..common.utils import normalize_response, rprint
from .base import BaseAgent
from .parser import ToolCallParser
from .types import ToolCall, ReactAgentResponse
from .exceptions import ToolCallParseError, ToolExecutionError
from .history import History
from .scratchpad import REACT_AGENT_PROMPT, REPAIR_ACTION_PROMPT, ANALYSIS_ONLY_MODE, DELEGATE_FINAL_MODEL
from .final_answer_generator import FinalAnswerGenerator
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
        id: str = "react_agent",
        llm: BaseChatModel = None,
        toolkit: Optional[Toolkit] = None,
        *,
        specific_instructions: str = "",
        config: Optional[ReActConfig] = None,
        logger: logging.Logger = logging.getLogger(__name__),
        tracer: Optional[Tracer] = None,
        log_traces: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.id = id
        self.llm = llm
        self.llm.silent()
        self.toolkit = toolkit
        self.specific_instructions = specific_instructions
        self.config = config or ReActConfig()
        self.logger = logger
        self.log_internal_thoughts = self.config.log_internal_thoughts
        self.log_traces = log_traces
        if self.llm is None:
            raise ValueError("llm must be provided")

         # Tracing setup
        # Base tracer that actually records and log steps (RichTracer by default).
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "react_agent",
                "agent_config": self.config,
                "model": self.llm.model_name,
                "toolkit": toolkit is not None,
                "log_steps": self.log_traces,
                "logging": "full" if self.log_internal_thoughts else "basic",
            },
            logger_=logger,
        )
        self.parser = ToolCallParser()
        self.memory = AgentMemory()
        self.tool_calls = []
        self._last_error: Optional[str] = None
        
        self.final_answer_generator = FinalAnswerGenerator(
            llm=self.llm, 
            language=self.config.final_answer_language,
            answer_length=self.config.final_answer_length,
            max_history_chars=self.config.final_answer_max_history_chars,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            logger=self.logger
        )

    # ---------- Public API ----------
    def run(
        self,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        query: Optional[str] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        specific_instructions: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        _route_extra_instructions: str = "",
        final_target_language: Optional[str] = None,
        final_answer_length: Optional[str] = None,
        final_answer_instructions: Optional[str] = None,
        reset_tracer: bool = True,
        return_traces: bool = True,
    ) -> ReactAgentResponse:
        """
        Run the ReAct agent.
        - If stream=True: returns ReactAgentResponse with a Generator[str]
          that yields tokens as they are produced.
        - If stream=False: returns ReactAgentResponse with the full string
          (we still use streaming under the hood, but we buffer it).
        """
        self._reset(reset_tracer)

        # Resolve config defaults
        user_prompt = (user_prompt if user_prompt is not None else query) or ""
        stream = self.config.return_stream_by_default if stream is None else bool(stream)
        temperature = float(self.config.temperature if temperature is None else temperature)
        max_new_tokens = int(self.config.max_new_tokens if max_new_tokens is None else max_new_tokens)
        specific_instructions = (
            self.specific_instructions
            if specific_instructions is None
            else specific_instructions or ""
        )

        stream_response = self._run_loop(
            query=user_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=stream,
            specific_instructions=specific_instructions,
            final_target_language=final_target_language,
            final_answer_length=final_answer_length,
            final_answer_instructions=final_answer_instructions,
        )
        # -------- Streaming path --------
        def no_stream(agent_response: Generator[str, None, None]) -> str:
            final_response = ""
            for chunk in agent_response:
                final_response += chunk
            return final_response
        
        return ReactAgentResponse(
            response=stream_response if stream else no_stream(stream_response), 
            traces=self.tracer.results if return_traces else None, 
            tool_calls=self.tool_calls
        )

    # ---------- Core loop ----------
    def _run_loop(
        self,
        query: str, 
        temperature: float, 
        max_new_tokens: int,
        stream: bool,
        specific_instructions: str,
        final_target_language: Optional[str] = None,
        final_answer_length: Optional[str] = None,
        final_answer_instructions: Optional[str] = None,
    ) -> Generator[str, None, str]:

        indent_level = max(self.tracer._depth - 1, 0)
        prefix = " " * (indent_level * 4)
        # prefix = indent + "    "

        rprint(prefix + "🧠 Thinking...", color="yellow")
        if self.log_traces:
            rprint(f"\n{prefix}[{self.id}] Tracing Start!")
        system_prompt = self._system_prompt(specific_instructions)
        main_tracer = self.tracer(
            agent_id=self.id,
            kind="react_agent",
            label="react_agent.run",
            inputs={
                "agent_type": type(self).__name__,
                "has_toolkit": bool(self.toolkit),
                "response_type": "streaming" if stream else "non-streaming",
                "specific_instructions": specific_instructions,
                "scratchpad": system_prompt,
            },  
        ).start()

        reason_tracer: Optional[TraceStepContext] = None 
        tool_tracer: Optional[TraceStepContext] = None

        history = History()
        self.stop_event = False
        iteration = 0
        final_answer = ""
        while not self.stop_event or iteration >= self.config.max_loop_iterations:
            # set persistent memory for the history
            self.set_persistent_memory(history=history.to_prompt())
            reasoning_prompt = self._build_prompt(query, history)
            # print(f"\n\nReasoning Prompt (query, memory, history):\n{reasoning_prompt}\n\n")
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

            response, thoughts = "", ""
            thought_started = False
            for chunk in streaming_response:
                if chunk.choices[0].finish_reason == "stop":
                    break
                part = chunk.choices[0].delta.content or ""
                response += part

                # Optional: stream internal thoughts to caller if enabled
                if self.config.return_internal_thoughts:
                    # very simple: just stream everything after "Thought:" once it appears
                    to_emit = ""
                    if not thought_started and "Thought:" in response:
                        thought_started = True
                        _prefix, suffix = response.split("Thought:", 1)
                        to_emit = suffix
                    elif thought_started:
                        to_emit = part

                    if to_emit:
                        if not self.config.skip_special_tokens:
                            to_emit = self.delims.sot + to_emit
                        yield to_emit

                thoughts += part
                if self.log_traces and self.log_internal_thoughts:
                    reason_tracer.stream(part, type="thought")
            
            reason_tracer.outputs(final_answer=final_answer, thoughts=thoughts)
            tool_call = self._decide_tool_call(response, query, history)
            # ---------- No tool decided ----------
            if tool_call is None or tool_call.tool is None:
                history.append(response)
                # In analysis_only mode: DO NOT synthesize a big final answer,
                # just say that analysis is done and return control to the caller.
                if self.config.mode == "analysis_only":
                    final_answer = (
                        "Analysis steps are complete; the latest tool results and memory "
                        "slots contain everything needed to generate the final user-facing answer."
                    )
                    reason_tracer.stream("[analysis_only] No further tool selected; returning control to parent agent.\n", type="info")
                    if not self.config.skip_special_tokens:
                        final_answer = self.delims.sof + final_answer + self.delims.eof
                    yield final_answer
                    break

                final_answer = ""
                reason_tracer.log(message=f"Generating final answer with language={final_target_language} length={final_answer_length} history_len={len(history)}", type="info")
                reason_tracer.stream("\nFinal Response:\n")
                if not self.config.skip_special_tokens:
                    yield self.delims.sof

                for chunk in self._generate_final_answer(
                    user_query=query,
                    history=history,
                    final_target_language=final_target_language,
                    final_answer_length=final_answer_length,
                    final_answer_instructions=final_answer_instructions,
                ):
                    reason_tracer.stream(chunk, type="whiteb")
                    yield chunk
                    final_answer += chunk
                if not self.config.skip_special_tokens:
                    yield self.delims.eof
                break
            
            reason_tracer.log(message=f"\n[🔧] Tool Selected: {tool_call.tool}", type="info", type_keyword=False)
            tool_tracer = self.tracer(
                agent_id=self.id,
                kind="tool.execute",
                label="react_agent.loop.tool_execute",
            ).start()

            history.append(response)
            # Execute tool safely with bounded retries
            # core.py, inside _run_loop, replacing the "execute tool" part
            tool_response = self._try_execute_tool(tool_call, tool_tracer)
            tool_results = normalize_response(tool_response.results)

            if isinstance(tool_results, GeneratorType):
                # live stream to the user and accumulate
                results_text = ""
                if tool_response.final_answer:
                    tool_tracer.stream("\nFinal Tool Result:\n")
                    if not self.config.skip_special_tokens:
                        yield self.delims.sof
                    for chunk in tool_results:
                        yield chunk
                        tool_tracer.stream(chunk, type="whiteb")
                        results_text += chunk
                    if not self.config.skip_special_tokens:
                        yield self.delims.eof
                    final_answer = results_text
                    break                
                else:
                    for chunk in tool_results:
                        results_text += chunk
                        # tool_tracer.stream(chunk, type="muted")

                history.append(f"results: {results_text}")
            else:
                # plain string
                results_text = tool_results
                if tool_response.final_answer:
                    final_answer = results_text
                    if not self.config.skip_special_tokens:
                        final_answer = self.delims.sof + results_text + self.delims.eof
                    yield final_answer
                    tool_tracer.log("Final Tool Result:\n", type_keyword=False)
                    tool_tracer.log(final_answer, type="whiteb", type_keyword=False)
                    break
                history.append(f"results: {results_text}")

            # update memory here to handle lazy loading extras
            self.memory.clear_ephemeral()
            # Update memory from extras, including metadata              
            self._update_memory_from_extras(tool_response.extras, scope="ephemeral", created_by=tool_call.tool)
            tool_tracer.outputs(memory_update=tool_response.extras)

            # update tool call
            tool_call.output = results_text
            self.tool_calls.append(tool_call)
            
            tool_tracer.outputs(tool_return=results_text)
            tool_tracer.log(f"\n[🔧] Tool Result: {results_text[:2000]}...", type="info", type_keyword=False)
            tool_tracer.close()
            reason_tracer.close()
            iteration += 1

        if tool_tracer and not tool_tracer.is_closed(): tool_tracer.close()
        if reason_tracer and not reason_tracer.is_closed(): reason_tracer.close()
        if main_tracer and not main_tracer.is_closed():
            main_tracer.outputs(final_answer=final_answer, total_iterations=iteration)
            main_tracer.close()
        if self.log_traces:
            rprint(f"\n{prefix}[{self.id}] Tracing End!\n")
        
        self._sort_tracing_results()  # sort tracing restuls
        return final_answer or "I couldn't determine the answer."

    def _reset(self, reset_tracer: bool = True):
        if reset_tracer:
            self.tracer.reset()
        self.tool_calls = []
        self._last_error = None
        self.memory.clear_ephemeral()

    def stop_generation(self):
        self.logger.info("[ReActAgent] Stopping generation...")
        try:
            self.llm.stop_generation()
        finally:
            super().stop_generation()

    def update_toolkit(self, toolkit: Toolkit) -> None:
        self.toolkit = toolkit

    def set_persistent_memory(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.memory.set_persistent(key, value)
            
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
            if dropped:
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

    def _generate_final_answer(
        self,
        user_query: str,
        history: History,
        *,
        final_target_language: Optional[str] = None,
        final_answer_length: Optional[str] = None,
        final_answer_instructions: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        When the reasoning model signals that it's done (no Action),
        we call the FinalAnswerGenerator to produce the user-facing answer.
        """
        history_text = history.to_prompt()
        if not self.final_answer_generator:
            # Fallback: old simple behavior if for some reason it's missing
            prompt = (
                f"# User Query:\n{user_query}\n"
                f"{history_text}"
                "\n# Next Steps:\nProduce a complete final answer now in one message."
            )
            response = self.llm.ask(
                user_prompt=prompt,
                system_prompt="You finalize answers succinctly and helpfully.",
                chat_history=[],
                temperature=max(0.2, self.config.temperature - 0.3),
                max_new_tokens=min(self.config.max_new_tokens, 1200),
                stream=True,
            )
            return normalize_response(response)
        return self.final_answer_generator.generate(
            user_query=user_query,
            history=history_text,
            memory=self.memory,
            target_language=final_target_language,
            answer_length=final_answer_length,
            extra_instructions=final_answer_instructions,
        )
        
    def _try_execute_tool(self, tool_call: ToolCall, tool_tracer: TraceStepContext) -> ToolResponse:
        tool_name = tool_call.tool
        tool = self.toolkit.registry[tool_name]

        attempts = 0
        last_err = None
        while attempts <= self.config.retry.max_tool_errors:
            try:
                tool_tracer.log(f"[🔧] Executing Tool: {tool_name}, Attempt: {attempts}, [📤] Inputs: {str(tool_call.inputs)[:200]}...", type="info")
                # NEW: extract any requested memory keys
                inputs = dict(tool_call.inputs)

                # keys requested by the LLM
                requested_keys: List[str] = []
                if tool_call.memory_keys:
                    if isinstance(tool_call.memory_keys, str):
                        requested_keys = [tool_call.memory_keys]
                    elif isinstance(tool_call.memory_keys, list):
                        requested_keys = [str(k) for k in tool_call.memory_keys if k is not None]

                # keys forced by config for this tool    
                forced_keys: List[str] = self.config.forced_memory_keys.get(tool_name, []) or []

                # union, preserving order and uniqueness
                merged_keys: List[str] = []
                seen = set()
                for k in requested_keys + forced_keys:
                    if k not in seen:
                        seen.add(k)
                        merged_keys.append(k)

                mem_injected: Dict[str, Any] = {}
                if merged_keys:
                    # You can keep using your memory API; just be tolerant of missing keys
                    try:
                        mem_injected = self.memory.resolve_keys(merged_keys)
                    except Exception:
                        # If resolve_keys blows up on unknown keys, fall back to
                        # a best-effort manual gather.
                        mem_injected = {}
                        epi = self.memory.get_memory(mode="ephemeral") or {}
                        for k in merged_keys:
                            if k in epi:
                                mem_injected[k] = epi[k]
                
                # print("\n\nMemory keys (requested): ", requested_keys)
                # print("Forced memory keys: ", forced_keys)
                # print("Merged memory keys: ", merged_keys)
                # print("\nMemory injected (raw): ", mem_injected)
                # print("\n\nMemory: ", self.memory.get_memory(mode="ephemeral"))

                # Normal runtime memory (if you still want a global fallback)
                # mem_snapshot = self.memory.snapshot_for_tool()
                persistent_memory = self.memory.get_memory(mode="persistent")
                all_inputs = {
                    **inputs,
                    **persistent_memory, # optional: global injection
                    "context": mem_injected,      # memory selected by LLM
                }
                tool_tracer.inputs(
                    llm_generated_inputs=tool_call.inputs,
                    persistent_memory=persistent_memory,
                    mem_injected=mem_injected
                )
                tool_response: ToolResponse = tool(**all_inputs)
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
        tool_descriptions = self.toolkit.get_tools_description() if self.toolkit else "No tools."
        # mode_instructions = ANALYSIS_ONLY_MODE if self.config.mode == "analysis_only" else DELEGATE_FINAL_MODEL
        return REACT_AGENT_PROMPT.format(
            tool_descriptions=tool_descriptions,
            specific_instructions=specific_instructions,
        )

    def _build_prompt(self, user_query: str, history: History) -> str:
        prompt = f"# User Query:\n{user_query}\n"
        prompt += history.to_prompt()

        # NEW: memory listing for the LLM
        memory_block = self.memory.llm_visible_summary(mode="ephemeral", return_json=True)
        prompt += (
            "\n# Working Memory:\n"
            "You have access to the following memory slots from previous tool calls.\n"
            "Each slot can be passed to a future tool by referencing its `key`.\n"
            f"{memory_block}\n\n"
        )

        prompt += (
            "# Next Steps: (What should you do next for this step?)\n"
            "You are continuing the reasoning from the previous steps above.\n"
            "Thought: <your thoughts here>\n"
            "Action: <your action here> (JSON if tool call, None if necessary work has been done)\n"
        )
        return prompt

    def _sort_tracing_results(self):
        self.tracer.results.steps = sorted(self.tracer.results.steps, key=lambda s: s.step_id)

    def _update_memory_from_extras(
        self,
        extras: Dict[str, Any],
        scope: str = "ephemeral",
        created_by: Optional[str] = None,
    ) -> None:
        """
        Generic adapter: take `ToolResponse.extras` and populate AgentMemory.

        Convention for rich slots (tool-controlled):

            extras["some_key"] = {
                "value": <any python object / JSON-ish>,
                "description": "short human description",
                "visible_to_llm": True/False,        # default: False
                # optional:
                # "scope": "ephemeral" | "persistent"
                # "created_by": "<tool_name>"
            }

        Anything that is NOT of this shape is treated as a raw runtime value
        (stored but not visible to the LLM by default).
        """
        if not extras:
            return

        for key, raw in extras.items():
            # Defaults
            slot_scope = scope
            slot_created_by = created_by

            if isinstance(raw, dict) and "value" in raw:
                value = raw["value"]
                description = str(raw.get("description", "")).strip()
                visible_to_llm = bool(raw.get("visible_to_llm", False))

                # Allow per-slot overrides
                if "scope" in raw:
                    slot_scope = str(raw["scope"])
                if raw.get("created_by"):
                    slot_created_by = str(raw["created_by"])
            else:
                # Bare value → runtime-only, hidden from LLM by default
                value = raw
                description = ""
                visible_to_llm = False

            if slot_scope == "persistent":
                self.memory.set_persistent(
                    key=key,
                    value=value,
                    description=description,
                    visible_to_llm=visible_to_llm,
                    created_by=slot_created_by,
                )
            else:
                self.memory.set_ephemeral(
                    key=key,
                    value=value,
                    description=description,
                    visible_to_llm=visible_to_llm,
                    created_by=slot_created_by,
                )