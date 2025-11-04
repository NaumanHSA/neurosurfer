from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import ast
from pathlib import Path
import subprocess
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from neurosurfer.models.chat_models.base import BaseModel
from neurosurfer.agents.rag import RAGAgent

@dataclass
class CodegenRequest:
    name: str                      # e.g. "calculator"
    module_path: str               # e.g. "neurosurf/tools/builtin/calculator.py"
    purpose: str                   # human description of the tool
    when_to_use: str               # short guidance for agents
    inputs: List[Dict[str, Any]]   # [{name,type,description,required}]
    returns: Dict[str, str]        # {"type": "...", "description": "..."}

LLM_SYSTEM = """You are a senior Neurosurf maintainer.
Generate a NEW Python tool module that conforms to Neurosurf's Tool contract:
- Must subclass BaseTool and define a valid ToolSpec (name, description, when_to_use, inputs, returns).
- __call__(...) must validate logic and return ToolResponse(final_answer: bool, observation: str|Generator, extras: dict).
- Keep the public API minimal; no external deps beyond stdlib and Neurosurf's own modules.
- The code must be deterministic, safe (no network, no file writes), and easy to test.
- Include clear error messages for invalid inputs.
"""

LLM_USER_TEMPLATE = """Repository context (summarized):
{repo_context}

Generate a new tool:

- Name: {name}
- Module path: {module_path}
- Purpose: {purpose}
- When to use: {when_to_use}
- Inputs:
{inputs_block}
- Returns: {returns_type} — {returns_desc}

Constraints:
- Implement a class `{class_name}` in a single file {module_path}.
- Follow existing style (imports, dataclasses optional, minimal helpers).
- Do NOT write tests or docs here; only the tool module.
- Output ONLY the Python source code for the module (no fences).
"""

class CodegenTool(BaseTool):
    spec = ToolSpec(
        name="codegen_tool",
        description="Generate a new Neurosurf tool module using repo-aware RAG + LLM (no templates).",
        when_to_use="When you need to implement a new tool from scratch that conforms to BaseTool/ToolSpec.",
        inputs=[
            ToolParam(name="name", type="string", description="Tool name (slug)", required=True),
            ToolParam(name="module_path", type="string", description="Repo-relative .py path to write", required=True),
            ToolParam(name="purpose", type="string", description="What the tool does", required=True),
            ToolParam(name="when_to_use", type="string", description="Guidance message for agents", required=True),
            ToolParam(name="inputs", type="array", description="List of {name,type,description,required}", required=True),
            ToolParam(name="returns", type="object", description="Return {type,description}", required=True),
        ],
        returns=ToolReturn(type="string", description="Short result status or error message"),
    )

    def __init__(self, llm: BaseModel, rag: RAGAgent, temperature: float = 0.7):
        """Initialize with LLM and RAG retriever."""
        self.llm = llm
        self.rag = rag
        self.temperature = temperature

    def __call__(
        self,
        *,
        name: str,
        module_path: str,
        purpose: str,
        when_to_use: str,
        inputs: List[Dict[str, Any]],
        returns: Dict[str, str],
        **kwargs
    ) -> ToolResponse:
        try:
            # 1) Pull repo contract context via RAG
            context_query = "Neurosurf tool contract: BaseTool, ToolResponse, ToolSpec, ToolParam, ToolReturn; examples and style."
            res = self.rag.retrieve(
                user_query=context_query,
                base_system_prompt="Summarize contract and examples needed to implement a new tool.",
                base_user_prompt="Use this context to build a compact summary:\n{context}\n\nReturn a concise, precise contract.",
                chat_history=[]
            )
            contract_summary = self.llm.ask(
                user_prompt=res.base_user_prompt.replace("{context}", res.context),
                system_prompt=res.base_system_prompt,
                max_new_tokens=res.max_new_tokens,
                temperature=self.temperature
            ).choices[0].message.content

            # 2) Build prompt for codegen
            inputs_block = "\n".join(
                [f"  - {p['name']}: {p['type']} — {p.get('description','')} (required={bool(p.get('required', True))})"
                 for p in inputs]
            )
            class_name = "".join([s.capitalize() for s in name.split("_")]) + "Tool"
            user_prompt = LLM_USER_TEMPLATE.format(
                repo_context=contract_summary,
                name=name,
                module_path=module_path,
                purpose=purpose,
                when_to_use=when_to_use,
                inputs_block=inputs_block,
                returns_type=returns.get("type", "object"),
                returns_desc=returns.get("description", ""),
                class_name=class_name
            )

            # 3) Generate code with LLM
            code_resp = self.llm.ask(
                user_prompt=user_prompt,
                system_prompt=LLM_SYSTEM,
                max_new_tokens=3000,
                temperature=self.temperature
            ).choices[0].message.content

            # 4) Quick validation: parse AST
            try:
                ast.parse(code_resp)
            except SyntaxError as e:
                return ToolResponse(final_answer=False, observation=f"Generated code has syntax error: {e}")

            # 5) Write file
            path = Path(module_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code_resp, encoding="utf-8")

            # 6) Optional: auto-format/lint (best-effort)
            try:
                subprocess.run(["ruff", "check", "--fix", str(path)], capture_output=True, text=True)
            except Exception:
                pass

            return ToolResponse(final_answer=False, observation=f"Generated and wrote: {module_path}")
        except Exception as e:
            return ToolResponse(final_answer=False, observation=f"llm_codegen_tool error: {e}")
