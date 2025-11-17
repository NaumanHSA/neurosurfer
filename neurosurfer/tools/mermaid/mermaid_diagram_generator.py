from dataclasses import field
from typing import Optional, Dict, List, Any, Union, Generator
import logging

from .base_tool import BaseTool 
from ..models.embedders.base import BaseEmbedder  
from ..models.chat_models.base import BaseChatModel
from ..utils.prompts import MERMAID_DIAGRAM_GENERATION_PROMPT


class MermaidDiagramGenerator(BaseTool):
    name: str = "mermaid_diagram_generator"
    description: str = """This tool should be used **only** to generate valid Mermaid diagram code from a **natural language description** (e.g., "describe a flowchart of a login process"). It does not fix broken or invalid Mermaid code. If the user provides Mermaid-like syntax with errors or asks for syntax fixing, you must use `general_query_assistant` instead."""
    register: bool = True
    stream: bool = True
    
    def __init__(
        self,
        llm: BaseChatModel,
        embedder: BaseEmbedder,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(llm, embedder, logger)
        self.top_k: int = 10
        self.prompt = MERMAID_DIAGRAM_GENERATION_PROMPT
        self.special_token: str = " [__MERMAID__] "

    def __call__(
        self,
        user_query: str,
        chat_history: List[dict] = [],
        **kwargs: Any,
    ) -> Optional[Union[dict, Generator[str, None, None]]]:
        """Generate Mermaid diagram code using the LLM."""
        session_id = kwargs.get("session_id", None)
        vectorstore = kwargs.get("vectorstore", None)
        project_tree = kwargs.get("project_tree", None)    
        stream = kwargs.get("stream", self.stream)    

        if project_tree:            
            system_prompt, user_prompt, max_new_tokens = self.get_architecture_context(
                user_query=user_query,
                prompt=self.prompt,
                session_id=session_id,
                project_tree=project_tree,
                vectorstore=vectorstore,
                chat_history=chat_history,
                top_k=self.top_k,
            )
        else:
            system_prompt, user_prompt, max_new_tokens = self.build_db_context(
                session_id=session_id,
                user_query=user_query,
                prompt=self.prompt,
                vectorstore=vectorstore,
                top_k=self.top_k,
                chat_history=chat_history,
            )
            
        mermaid_code = ""
        response = self.llm_call(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            temperature=self.temperature,
            max_new_tokens=max_new_tokens,
            stream=stream
        )
        mermaid_code = ''
        if stream:
            for chunk in response:
                mermaid_code += chunk['choices'][0]['message']['content']
                yield chunk
        else:
            mermaid_code = response['choices'][0]['message']['content']
            yield response

        mermaid_code = mermaid_code.replace("```", "").replace("mermaid", "").strip()
        # Optional: Send to Mermaid renderer and return base64 image
        # base64 = mermaid_api(mermaid_code)
        # yield base64

    def __str__(self):
        return f"{self.name}: {self.description}"