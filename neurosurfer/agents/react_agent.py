"""
ReAct Agent Module
==================

This module implements the ReAct (Reasoning + Acting) agent pattern for Neurosurfer.
The ReAct agent combines reasoning and action in an iterative loop, allowing it to
break down complex tasks, use tools, and refine its approach based on observations.

The ReActAgent:
    - Thinks step-by-step about the user's query
    - Decides which tools to use and when
    - Executes tools and observes results
    - Refines its strategy based on observations
    - Generates a final answer when ready

Key Features:
    - Streaming support for real-time output
    - Tool execution with automatic input validation
    - Memory system for passing data between tool calls
    - Error handling and recovery
    - Stop signal support for interrupting generation

The agent follows the ReAct pattern:
    Thought -> Action -> Observation -> Thought -> ... -> Final Answer
"""
import json
import re
import time
import traceback
from typing import Dict, List, Any, Union, Generator, Optional
from rich import print
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from ..models.chat_models.base import BaseModel
from ..tools import Toolkit
from ..tools.base_tool import ToolResponse


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent for complex task solving.
    
    This agent uses an iterative reasoning loop to solve complex queries by:
    1. Analyzing the user's question
    2. Deciding which tool(s) to use
    3. Executing tools and observing results
    4. Refining its approach based on observations
    5. Generating a final answer
    
    The agent maintains a conversation history and can handle multi-step
    reasoning tasks that require multiple tool invocations.
    
    Attributes:
        toolkit (Toolkit): Collection of available tools
        llm (BaseModel): Language model for reasoning
        logger (logging.Logger): Logger for debugging
        verbose (bool): Enable verbose output
        specific_instructions (str): Additional instructions for the agent
        sof (str): Start-of-final-answer marker
        eof (str): End-of-final-answer marker
        memory_items (dict): Temporary memory for passing data between tool calls
        stop_event (bool): Flag to stop generation
    
    Example:
        >>> from neurosurfer.agents import ReActAgent
        >>> from neurosurfer.tools import Toolkit
        >>> from neurosurfer.models.chat_models import TransformersModel
        >>> 
        >>> llm = TransformersModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
        >>> toolkit = Toolkit()
        >>> toolkit.register_tool(MyTool())
        >>> 
        >>> agent = ReActAgent(toolkit=toolkit, llm=llm, verbose=True)
        >>> 
        >>> # Non-streaming
        >>> for chunk in agent.run("What is the weather in Paris?"):
        ...     print(chunk, end="")
    """
    def __init__(
        self,
        toolkit: Toolkit,
        llm: BaseModel,
        logger: logging.Logger = logging.getLogger(),
        verbose: bool = True,
        specific_instructions: str = ""
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            toolkit (Toolkit): Collection of tools available to the agent
            llm (BaseModel): Language model for reasoning and generation
            logger (logging.Logger): Logger instance for debugging. Default: root logger
            verbose (bool): Enable verbose output (tool calls, observations). Default: True
            specific_instructions (str): Additional instructions to append to the system prompt.
                Use this to customize agent behavior. Default: ""
        
        Example:
            >>> toolkit = Toolkit()
            >>> llm = TransformersModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
            >>> agent = ReActAgent(
            ...     toolkit=toolkit,
            ...     llm=llm,
            ...     verbose=True,
            ...     specific_instructions="Always be concise in your answers."
            ... )
        """
        self.toolkit = toolkit
        self.llm = llm
        self.logger = logger
        self.verbose = verbose
        self.specific_instructions = specific_instructions
        self.schema_context = ""
        self.raw_results = ""
        self.sof = "<__final_answer__>" # start of final answer
        self.eof = "</__final_answer__>" # end of final answer
        self.memory_items = dict()
        self.stop_event = False

    def run(self, user_query: str, **kwargs: Any) -> Generator:
        """
        Execute the agent on a user query.
        
        This is the main entry point for running the ReAct agent. It yields
        streaming output as the agent reasons, executes tools, and generates
        the final answer.
        
        Args:
            user_query (str): The user's question or task
            **kwargs: Additional generation parameters:
                - temperature (float): Sampling temperature. Default: 0.7
                - max_new_tokens (int): Max tokens to generate. Default: 8000
                - Other model-specific parameters
        
        Yields:
            str: Streaming text chunks including thoughts, tool calls, observations,
                and the final answer
        
        Returns:
            str: The final answer (also yielded in the stream)
        
        Example:
            >>> agent = ReActAgent(toolkit=toolkit, llm=llm)
            >>> for chunk in agent.run("What is 2+2?", temperature=0.5):
            ...     print(chunk, end="")
            Thought: This is a simple math question...
            <__final_answer__>The answer is 4</__final_answer__>
        """
        return self.run_agent__(user_query, **kwargs)

    def run_agent__(self, user_query: str, **kwargs: Any) -> Generator:
        """
        Main reasoning loop for the ReAct agent.
        
        This method implements the core ReAct pattern:
        1. Generate reasoning with LLM
        2. Parse tool calls from reasoning
        3. Execute tools and observe results
        4. Add observations to history
        5. Repeat until final answer is generated
        
        The method handles:
        - Streaming LLM responses
        - Tool call parsing and validation
        - Error handling and recovery
        - Final answer detection and extraction
        - Stop signal enforcement
        
        Args:
            user_query (str): The user's question or task
            **kwargs: Generation parameters (temperature, max_new_tokens, etc.)
        
        Yields:
            str: Streaming text chunks
        
        Returns:
            str: Final answer or error message
        """
        history: List[str] = []
        system_prompt = self.react_system_prompt()
        final_answer = ""
        self.stop_event = False
        while not self.stop_event:
            reasoning_prompt = self.build_prompt(user_query, history)
            yield "\n\n[🧠] Chain of Thoughts...\n"
            stream = self.llm.ask(
                user_prompt=reasoning_prompt,
                system_prompt=system_prompt,
                chat_history=[],  # optionally pass history
                temperature=kwargs.get("temperature", 0.7),
                max_new_tokens=kwargs.get("max_new_tokens", 8000),
                stream=True
            )
            response, final_answer_started = "", False
            for chunk in stream:
                chunk_text = chunk["choices"][0].get("message", {}).get("content", "")
                response += chunk_text
                if not final_answer_started and self.sof in response:
                    final_answer_started = True
                    prefix, suffix = response.split(self.sof, 1)
                    if suffix.strip():
                        yield self.sof + suffix
                        final_answer += suffix
                elif final_answer_started:
                    yield chunk_text if self.eof not in chunk_text else chunk_text.split(self.eof)[0]
                    final_answer += chunk_text
                else:
                    yield chunk_text

            if final_answer_started:
                yield self.eof
                break

            if "Action:" not in response:
                yield "\n\n[❌] Error: No action found in response."
                break

            history.append(response)
            tool_call = self.parse_tool_call(response)
            # print("Tool Call Parse:", tool_call)  
            if not tool_call: break
            try:
                if self.verbose:
                    print(f"\n[🔧] Tool: {tool_call.get('tool')}\n[📤] Inputs: {tool_call.get('inputs')}")
                
                tool_response: ToolResponse = self.execute_llm_tool_output(tool_call, **kwargs)
                handled_tool_response = self._handle_tool_response(tool_response)
                if isinstance(handled_tool_response, Generator):
                    observation = ""
                    if tool_response.final_answer: yield self.sof
                    for chunk in handled_tool_response:
                        observation += chunk
                        if tool_response.final_answer:
                            yield chunk
                else:
                    observation = handled_tool_response

                if tool_response.final_answer:
                    yield self.eof
                    final_answer = observation
                    break

                history.append(f"Observation: {observation}")
                if self.verbose: 
                    print(f"Observation: {observation}")
            except Exception as e:
                error_msg = f"ERROR - {str(e)}"
                history.append(f"Observation: {error_msg}")
                print("\n[❌] Error:", str(e))
                print("Traceback Error:", traceback.format_exc())
                break

        self.logger.info(f"[ReActAgent] Stopped -> Final answer: {final_answer}")
        return final_answer or "I couldn't determine the answer."

    def _handle_tool_response_stream(self, tool_response: ToolResponse) -> Generator:
        prefix = "Final Answer: " if tool_response.final_answer else ""
        yield prefix
        for chunk in tool_response.observation:
            if isinstance(chunk, dict) and "choices" in chunk:
                yield chunk["choices"][0]["message"]["content"]
            else:
                yield chunk

    def _handle_tool_response_non_stream(self, tool_response: ToolResponse) -> str:
        prefix = "Final Answer: " if tool_response.final_answer else ""
        if isinstance(tool_response.observation, dict) and "choices" in tool_response["observation"]:
            return prefix + tool_response.observation["choices"][0]["message"]["content"]
        else:
            return prefix + str(tool_response.observation)

    def _handle_tool_response(self, tool_response: ToolResponse) -> Union[str, Generator]:
        """Handle a tool response, optionally yielding final answer if marked."""
        # add tool retuns other than final_answer and observation to memory_items
        for k, v in tool_response.extras.items():
            self.memory_items[k] = v
        stream = isinstance(tool_response.observation, Generator)
        return self._handle_tool_response_stream(tool_response) if stream else self._handle_tool_response_non_stream(tool_response)

    def build_prompt(self, user_query: str, history: List[str]) -> str:
        prompt = f"# User Query:\n{user_query}\n"        
        if history:
            prompt += "# Chain of Thoughts:\n"
            for h in history:
                prompt += f"{h}\n"
        prompt += "\n# Next Steps:\nWhat should you do next?\n If you think the answer is ready, generate a complete Final Answer independent of the history.\n"
        return prompt

    def extract_inputs_from_description(self,desc: str) -> List[str]:
        match = re.search(r"Inputs:\n((?:\s+-\s+\w+\s+\(.+?\):.+\n)+)", desc)
        if not match:
            return []
        return [line.strip().split()[1] for line in match.group(1).splitlines()]


    def parse_tool_call(self, text: str) -> Optional[dict]:
        """
        Parse tool call from LLM response.
        
        Extracts and validates tool calls from the agent's reasoning output.
        Tool calls must be in JSON format:
        Action: {"tool": "tool_name", "inputs": {...}, "final_answer": bool}
        
        Args:
            text (str): LLM response text containing potential tool call
        
        Returns:
            Optional[dict]: Parsed tool call with keys:
                - tool (str): Tool name
                - inputs (dict): Validated tool inputs
            Returns None if no valid tool call found
        
        Example:
            >>> text = 'Thought: I need data\nAction: {"tool": "sql_query", "inputs": {"query": "SELECT *"}}'
            >>> tool_call = agent.parse_tool_call(text)
            >>> print(tool_call)
            {'tool': 'sql_query', 'inputs': {'query': 'SELECT *'}}
        """
        match = re.search(r"Action:\s*({.*})", text, re.DOTALL)
        if not match:
            return None
        try:
            tool_call_raw = json.loads(match.group(1))
            tool_name = tool_call_raw.get("tool")
            tool_inputs = tool_call_raw.get("inputs", {})
            final_answer = tool_call_raw.get("final_answer", False)

            # Filter tool inputs using the tool's known schema
            tool = self.toolkit.registry.get(tool_name, None)
            if not tool:
                return None

            spec = self.toolkit.specs[tool_name]
            cleaned_inputs = spec.check_inputs(tool_inputs)  # strict: required, types, no extras
            # Pass final flag outside of schema (agent-level concern)
            cleaned_inputs["final_answer"] = final_answer
            return {
                "tool": tool_name,
                "inputs": cleaned_inputs,
            }
        except json.JSONDecodeError:
            return None

    def execute_llm_tool_output(self, tool_call: dict, **context_kwargs) -> ToolResponse:
        """
        Execute a tool with LLM-generated inputs and runtime context.
        
        This method:
        1. Retrieves the tool from the toolkit
        2. Merges LLM inputs with runtime context (llm, db_engine, etc.)
        3. Merges with memory items from previous tool calls
        4. Executes the tool
        5. Clears memory items after execution
        
        Args:
            tool_call (dict): Parsed tool call with keys:
                - tool (str): Tool name
                - inputs (dict): Tool inputs from LLM
            **context_kwargs: Runtime context injected by the agent:
                - llm: Language model instance
                - db_engine: Database connection
                - embedder: Embedding model
                - vectorstore: Vector database
                - etc.
        
        Returns:
            ToolResponse: Tool execution result
        
        Raises:
            ValueError: If tool is not registered in toolkit
        
        Example:
            >>> tool_call = {"tool": "sql_query", "inputs": {"query": "SELECT *"}}
            >>> response = agent.execute_llm_tool_output(
            ...     tool_call,
            ...     db_engine=engine,
            ...     llm=llm
            ... )
        """
        tool_name = tool_call["tool"]
        input_params = tool_call["inputs"]
        if tool_name not in self.toolkit.registry:
            raise ValueError(f"Tool '{tool_name}' is not registered.")
        tool = self.toolkit.registry[tool_name]
        all_inputs = {
            **input_params,     # LLM-generated inputs
            **context_kwargs,    # Runtime context like llm, db_engine, etc
            **self.memory_items  # Memory items from previous tool calls
        }
        tool_response = tool(**all_inputs)
        self.memory_items = dict()   # clear memory items after tool call
        return tool_response

    def react_system_prompt(self) -> str:
        tool_descriptions = self.toolkit.get_tools_description().strip()
        return REACT_AGENT_PROMPT.format(tool_descriptions=tool_descriptions, specific_instructions=self.specific_instructions)
    
    def update_toolkit(self, toolkit: Toolkit):
        self.toolkit = toolkit

    def stop_generation(self):
        """
        Stop the agent's generation immediately.
        
        Sets stop signals for both the LLM and the agent loop.
        This allows graceful interruption of long-running tasks.
        
        Example:
            >>> import threading
            >>> agent = ReActAgent(toolkit=toolkit, llm=llm)
            >>> 
            >>> # Run in thread
            >>> def run():
            ...     for chunk in agent.run("Long task..."):
            ...         print(chunk, end="")
            >>> 
            >>> thread = threading.Thread(target=run)
            >>> thread.start()
            >>> # ... later ...
            >>> agent.stop_generation()
        """
        self.logger.info("[ReActAgent] Stopping generation...")
        self.llm.stop_generation()
        self.stop_event = True


REACT_AGENT_PROMPT = """
You are a reasoning agent that answers user questions using a set of external tools.

## Goal:
Think step-by-step and use tools when necessary to find the answer. Do not make assumptions about any hidden variables — tools will be injected with the correct dependencies at runtime.

---

## Behavior Guidelines:

- 🔍 Think step-by-step.
- 📚 If you are confused about the user's question, ask for clarification.
- 🛠️ If a tool is needed, stop and call the most relevant tool using only its **explicit inputs**.
- 🚫 Do NOT include any additional parameters. Only pass what the tool requires.
- 🔁 Call only **one tool at a time**. After each `Observation`, re-evaluate your strategy.
- ✅ If a tool is needed, end your message with the Action block.
- 🧠 Refine the query for the tool which best describes the user's question. If an error or observation occurs, use it to refine your next action.
- ❌ Do NOT repeat the same tool call with the same inputs if it failed before.
- 📝 Rewrite the tool inputs with what you've learned (e.g., missing columns, filter conditions, formatting issues).
- ✅ If no tool is needed, respond directly.

---

## Tool Call Format:

```json
Action: {{
  "tool": "tool_name",
  "inputs": {{
    "param1": "value1",
    "param2": "value2"
  }},
  "final_answer": true  // true if this tool response should be treated as the final answer
}}
```

- Set `"final_answer": true` when you expect the tool output to be the final user-facing answer.
- Set `"final_answer": false` when you are only using the tool for intermediate reasoning.

---

## Response Format:
You must follow this reasoning format:

```
Thought: explain what you're thinking or trying to solve.
Action: {{
  "tool": "tool_name",
  "inputs": {{
    "param": "value"
  }},
  "final_answer": true  // true if this tool response should be treated as the final answer
}}
```
After a tool is used, your next input will begin with:

```
Observation: Response from the tool called
Thought: reason about the result
```

When you have a final answer, respond with:

```
Thought: reason about the result
<__final_answer__>Final Answer: the final response to the user.</__final_answer__>
```

Always think after each step and determine the next best action.
Avoid repeating words or reusing prompt examples verbatim.

---

## Available Tools:
{tool_descriptions}

---

{specific_instructions}
"""