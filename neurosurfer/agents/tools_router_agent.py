import json
import re
import logging
import traceback
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

from ..models.chat_models.base import BaseModel
from ..tools import Toolkit
from ..tools.base_tool import BaseTool, ToolResponse


class ToolsRouterAgent:
    def __init__(
        self,
        toolkit: Toolkit,
        llm: BaseModel,
        logger: logging.Logger = logging.getLogger(__name__),
        verbose: bool = False,
        specific_instructions: str = "",
    ):
        """
        toolkit: object with .registry: Dict[str, ToolInstance]
                 ToolInstance typically has: description (str), register (bool), and
                 one or more of: run(**kwargs)->str, stream(**kwargs)->Generator[str, None, None],
                 run_stream(**kwargs)->Generator[str, None, None], __call__(**kwargs)
        llm:     chat/generative model with a .generate(...) or .chat(...) method (see _route())
        """
        self.toolkit = toolkit
        self.llm = llm
        self.logger = logger
        self.verbose = verbose
        self.specific_instructions = specific_instructions

    # ---------- PUBLIC API ----------
    def run(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Decide a tool with the LLM (JSON: {"tool": "...", "inputs": {...}}),
        then execute that tool. The tool receives: query=<user_query>, plus router inputs,
        plus **kwargs (kwargs are included as-is).
        - If stream=True: returns a Generator[str, ...].
        - If stream=False: returns a str.
        """
        tool_name, tool_inputs = self._route(user_query, chat_history or [])
        print("Using tool: ", tool_name)
        print("Tool inputs: ", tool_inputs, "\n\n")
        if tool_name == "none" or not tool_name:
            return self._error_response(message=f"No suitable tool selected for the user query.\nUser query: {user_query}")

        tool: BaseTool = self.toolkit.registry.get(tool_name)
        if tool is None:
            return self._error_response(message=f"Selected tool '{tool_name}' is not registered.")

        # Merge payload (router inputs override kwargs), always include the query
        payload = {"query": user_query, **kwargs, **(tool_inputs or {})}
        return self._execute_tool(tool, **payload)

    # ---------- TOOL EXECUTION HELPERS ----------
    def _execute_tool(self, tool: BaseTool, **payload: Any) -> Union[str, Generator[str, None, None]]:
        try:
            response: ToolResponse = tool(**payload)
            if isinstance(response.observation, Generator):
                yield from response.observation
            else:
                yield response.observation
        except Exception as e:
            err = f"[Tool Response ERRRO] '{getattr(tool, 'name', str(tool))}' error: {traceback.format_exc()}"
            if self.verbose:
                self.logger.exception(err)
            yield from self._as_stream(err)

    # ---------- LLM ROUTING ----------
    def _route(
        self, 
        user_query: str,
        chat_history: List[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Ask the LLM to pick a tool. Returns (tool_name, inputs_dict).
        Expects a single-line JSON: {"tool": "...", "inputs": {...}}
        """
        system_prompt = self._tools_router_system_prompt()
        routing_prompt = self._format_router_input(user_query, chat_history)
        raw = self.llm.ask(
            user_prompt=routing_prompt,
            system_prompt=system_prompt,
            chat_history=[],
            temperature=kwargs.get("temperature", 0.7),
            max_new_tokens=kwargs.get("max_new_tokens", 4000),
            stream=False
        )["choices"][0]["message"]["content"]
        if self.verbose:
            self.logger.info(f"[router] raw routing output: {raw}")
        tool_name, inputs = self._extract_tool_json(raw)
        return tool_name, inputs

    # ---------- PROMPT BUILDING ----------
    def _tools_router_system_prompt(self) -> str:
        tool_descriptions = self.toolkit.get_tools_description().strip()
        return """
You are a stateless tool router. Given the user's message and the catalog of tools below, output a SINGLE JSON object with EXACTLY these two keys:
{{"tool": "<tool_name>", "inputs": {{<param>:<value>}}, "optimized_query": "<optimized and detailed query clearly explaining the user's message>"}}

STRICT RULES:
- Output MUST be valid JSON, on one line, with only the keys "tool" and "inputs".
- Do NOT include any other keys or text (no explanations, no code fences, no thoughts).
- Choose at most one tool.
- Use only explicit inputs defined in that tool's description. Do NOT invent parameters or values.
- Include only required parameters. Do not include optional parameters.
- The optimized value must clearly capture the meaning of the user’s request.  
- If the user asked multiple questions, break them down into structured text (paragraphs or bullet points) rather than a single run-on query.  
- If no tool is appropriate OR required inputs are missing/ambiguous, output: {{"tool":"none","inputs":{{}}}}
- If a clarifier tool 'ask_user' exists, you may return: {{"tool":"ask_user","inputs":{{"question":"<concise question>"}}}}

TOOL CATALOG:
{tool_descriptions}

{specific_instructions}
""".format(tool_descriptions=tool_descriptions, specific_instructions=self.specific_instructions).strip()

    def _format_router_input(self, user_query: str, chat_history: List[Dict[str, str]]) -> str:
        # Keep the router context minimal and deterministic.
        hist_lines = []
        for m in (chat_history or [])[-10:]:
            role = m.get("role", "user")
            content = m.get("content", "").replace("\n", " ").strip()
            hist_lines.append(f"{role}: {content}")
        history_block = "\n".join(hist_lines)
        return f"User message:\n{user_query}\n\nRecent chat history (optional, newest last):\n{history_block}"

    # ---------- UTILITIES ----------
    def _extract_tool_json(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Robustly parse {"tool": "...", "inputs": {...}} from LLM output.
        """
        # Remove code fences if any
        cleaned = re.sub(r"^```.*?\n|```$", "", text.strip(), flags=re.DOTALL)
        # Try direct parse first
        obj = self._try_json(cleaned)
        if self._looks_like_decision(obj):
            inputs = obj.get("inputs", {})
            if ("user_query" in inputs or "query" in inputs) and "optimized_query" in obj:
                inputs["user_query"] = obj.get("optimized_query")
            return obj["tool"], inputs

        # Fallback: find first JSON object substring
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            obj = self._try_json(match.group(0))
            if self._looks_like_decision(obj):
                inputs = obj.get("inputs", {})
                if "user_query" in inputs and "optimized_query" in obj:
                    inputs["user_query"] = obj.get("optimized_query")
                if "query" in inputs and "optimized_query" in obj:
                    inputs["query"] = obj.get("optimized_query")
                return obj["tool"], inputs
        # Give up -> none
        return "none", {}

    def _error_response(self, message: str) -> Dict[str, Any]:
        system_prompt = (
            "The pipeline failed to generate a valid response. "
            "Provide a concise and helpful answer to the user. "
            "Do not mention the failure. "
            "Ask the user to restate their question."
        )
        return self.llm.ask(
            system_prompt=system_prompt,
            user_prompt=f"User message:\n{message}",
            chat_history=[],
            temperature=0.7,
            max_new_tokens=2000,
            stream=True
        )


    @staticmethod
    def _looks_like_decision(obj: Any) -> bool:
        return isinstance(obj, dict) and "tool" in obj and "inputs" in obj and isinstance(obj["inputs"], dict)

    @staticmethod
    def _try_json(s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            return None

    @staticmethod
    def _stringify(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, (str, bytes)):
            return x.decode() if isinstance(x, bytes) else x
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    @staticmethod
    def _ensure_generator(x: Any) -> Generator[str, None, None]:
        if x is None:
            return
            yield  # pragma: no cover
        if isinstance(x, (list, tuple)):
            for item in x:
                yield str(item)
            return
        if isinstance(x, str):
            yield x
            return
        # If it's an iterable (but not str), iterate
        if isinstance(x, Iterable):
            for item in x:
                yield str(item)
            return
        # Fallback: yield stringified
        yield str(x)

    @staticmethod
    def _as_stream(s: str) -> Generator[str, None, None]:
        yield s
