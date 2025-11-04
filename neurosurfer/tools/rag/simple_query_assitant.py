"""
RagSimpleAnswerTool: single-prompt RAG answer (retrieval + synthesis)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ...models.chat_models.base import BaseModel
from ...agents.rag import RAGAgent
from ..base_tool import BaseTool, ToolResponse
from ..tool_spec import ToolSpec, ToolParam, ToolReturn


SIMPLE_QUERY_ASSISTANT_PROMPT = {
    "system_prompt": """You are a helpful assistant that answers a user's query based on context provided.""",
    "user_prompt": """Answer the following question:

User Question:
"{query}"

Context:
{context}

Return ONLY a valid answer.
"""
}

class RagSimpleAnswerTool(BaseTool):
    spec = ToolSpec(
        name="rag_simple_answer",
        description="Answer a question using the user's uploaded corpus (vector store).",
        when_to_use="Use when the user asks a question that can be answered using the user's uploaded corpus (vector store).",
        inputs=[
            ToolParam(name="query", type="string", description="the user's question.", required=True),
        ],
        returns=ToolReturn(type="string", description="Concise natural-language explanation."),
    )

    def __init__(
        self,
        llm: BaseModel,
        rag_agent: RAGAgent,
        stream: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.llm = llm
        self.stream = stream
        self.logger = logger
        self.rag_agent = rag_agent
        self.prompt = SIMPLE_QUERY_ASSISTANT_PROMPT
        self.top_k: int = 10
        self.max_new_tokens: int = 3000
        self.temperature: float = 0.3
        self.special_token: str = ""

    def __call__(
        self,
        query: str,
        **kwargs: Any,
    ) -> ToolResponse:
        top_k = kwargs.get("top_k", self.top_k)
        chat_history = kwargs.get("chat_history", [])
        
        retrieval_results = self.rag_agent.retrieve(
            user_query=query, 
            base_system_prompt=self.prompt["system_prompt"], 
            base_user_prompt=self.prompt["user_prompt"], 
            chat_history=chat_history, 
            top_k=top_k
        )
        response = self.llm.ask(
            system_prompt=self.prompt["system_prompt"],
            user_prompt=self.prompt["user_prompt"].format(query=query, context=retrieval_results.context),
            temperature=self.temperature,
            max_new_tokens=retrieval_results.max_new_tokens,
            stream=self.stream
        )
        return ToolResponse(final_answer=True, observation=response)

