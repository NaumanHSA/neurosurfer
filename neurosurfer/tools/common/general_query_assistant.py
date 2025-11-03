from __future__ import annotations

import logging
from typing import Any, Optional

from ...models.chat_models.base import BaseModel
from ...agents.rag_retriever_agent import RAGRetrieverAgent
from ..base_tool import BaseTool, ToolResponse
from ..tool_spec import ToolSpec, ToolParam, ToolReturn


GENERAL_QUERY_ASSISTANT_PROMPT = {
    "system_prompt": "You are a helpful AI assistant that answers general user queries clearly and concisely.",
    "user_prompt": """Answer the following user question as clearly and directly as possible:

User Question:
"{user_query}"

Return ONLY the answer, no extra text.
""",
}
# -----------------------------------------------------------------------------
# GeneralQueryAssistantTool
# -----------------------------------------------------------------------------
class GeneralQueryAssistantTool(BaseTool):
    spec = ToolSpec(
        name="general_query_assistant",
        description="Provide direct answers to general user queries that do not require retrieval from a vector store.",
        when_to_use="Use when the user asks a general question that does not require retrieval from a vector store.",
        inputs=[
            ToolParam(name="query", type="string", description="the user's question.", required=True),
        ],
        returns=ToolReturn(type="string", description="Concise natural-language explanation."),
    )

    def __init__(
        self,
        llm: BaseModel = None,
        stream: bool = True,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.llm = llm
        self.stream = stream
        self.logger = logger
        self.prompt = GENERAL_QUERY_ASSISTANT_PROMPT
        self.max_new_tokens: int = 2000
        self.temperature: float = 0.7
        self.special_token: str = ""

    def __call__(
        self, 
        query: str, 
        **kwargs: Any
    ) -> ToolResponse:
        """
        Execute the tool to answer a general user query.

        Args:
            query (str): The user's question.
            **kwargs: Runtime parameters (e.g., temperature, max_new_tokens).
        Returns:
            ToolResponse: {'final_answer': False, 'observation': str|Iterable}
        """
        max_new_tokens = RAGRetrieverAgent._calculate_max_new_tokens_prompts(
            llm=self.llm,
            system_prompt=self.prompt["system_prompt"],
            user_prompt=self.prompt["user_prompt"],
            chat_history=[],
        )
        response = self.llm.ask(
            system_prompt=self.prompt["system_prompt"],
            user_prompt=self.prompt["user_prompt"].format(user_query=query),
            temperature=kwargs.get("temperature", self.temperature),
            max_new_tokens=max_new_tokens,
            stream=kwargs.get("stream", self.stream),
        )
        return ToolResponse(final_answer=False, observation=response)
