from typing import Dict, Any, Optional, Union, Generator
import logging

from ...models.chat_models.base import BaseModel
from ...db.sql_schema_store import SQLSchemaStore
from ..base_tool import BaseTool, ToolResponse
from ..tool_spec import ToolSpec, ToolParam, ToolReturn


DATABASE_INSIGHT_PROMPT = {
    "system_prompt": """You are a senior database architect with deep understanding of relational database systems.
Your task is to answer high-level questions about a given database by analyzing its tables, their purposes, and relationships.

Be precise and structured. Avoid assumptions that are not supported by the input.
If you don't have enough information, say so clearly.""",
    "user_prompt": """You are given documentation describing the tables in a database. Use this information to answer the following question:

{query}

# Tables Summaries:
{table_summaries}

Respond with a clear and concise answer. Focus only on what can be inferred from the data above.
"""
}


class DBInsightsTool(BaseTool):
    spec = ToolSpec(
        name="database_insight_tool",
        description="Answers high-level or conceptual questions about the database as a whole the kinds of questions that cannot be solved by writing or executing a SQL query. "
        "This includes explaining the purpose of the database, the roles of different tables, how entities relate to one another, "
        "and what architectural or security implications may arise from those relationships.\n"
        "For example, analyzing how organizational hierarchy tables like `Departments` and `Designations` interact with access control tables such as `GroupAccessRequests`, "
        "and what risks could result from misconfigurations.\n",
        when_to_use="Use this tool when you need to answer high-level or conceptual questions about the database that cannot be solved by writing or executing a SQL query.",
        inputs=[
            ToolParam(name="query", type="str", description="A natural language question about the database's structure, design, or semantics.")
        ],
        returns=ToolReturn(type="str", description="A natural language explanation that synthesizes table metadata, relationships, and domain context into an insightful answer.")
    )
   
    def __init__(
        self,
        llm: BaseModel,
        sql_schema_store: SQLSchemaStore,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.llm = llm
        self.sql_schema_store = sql_schema_store
        self.logger = logger
        self.prompt = DATABASE_INSIGHT_PROMPT
        self.max_new_tokens: int = 3000
        self.temperature: float = 0.7
        self.stream = True
        self.final_answer: bool = True
        self.special_token: str = " [__DATABASE_INSIGHT__] "

    def __call__(
        self, 
        query: str, 
        **kwargs: Any,
    ) -> ToolResponse:
        final_answer = kwargs.get("final_answer", self.final_answer)

        # Construct LLM prompt
        table_summaries = self.get_tables_summaries__()
        system_prompt = self.prompt["system_prompt"]
        user_prompt = self.prompt["user_prompt"].format(
            query=query,
            table_summaries=table_summaries
        )
        response = self.llm.ask(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            stream=self.stream
        )
        return ToolResponse(final_answer=final_answer, results=response)
      
    def get_tables_summaries__(self):
        table_summaries = ""
        for table_name, table_data in self.sql_schema_store.store.items():
            summary = table_data["summary"]
            table_info = f"""Table: {table_name}\nSummary: {summary}"""
            table_summaries += f"{table_info}\n\n"
        return table_summaries