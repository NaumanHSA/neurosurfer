import logging
from typing import Optional, Dict, Any

from ...models.chat_models.base import BaseModel
from ..base_tool import BaseTool, ToolResponse
from ..tool_spec import ToolSpec, ToolParam, ToolReturn


SQL_QUERY_GENERATION_PROMPT = {
    "system_prompt": """You are an expert in writing optimized and syntactically correct SQL Server (MSSQL) queries.
Your task is to generate a valid SQL query based strictly on the user's input question and the provided schema.

Follow these strict rules:
- Do not use `*` while fetching results. Limit the results to avoid long generations unless explicitly asked (e.g. SELECT TOP 10 column_1 FROM table_1).
- Use `LIKE` instead of `=` for string comparisons.
- Always use `LIKE '%value%'` for partial string matches.
- Use `=` only for numeric or date values.
- For BIT columns, use 1 for TRUE and 0 for FALSE.
- Do not include explanations, comments, or additional formatting — only output the raw SQL query.
- If you don't know the answer, say so. Do not make up SQL queries unless the context is clear.

Do not include any explanation, commentary, or formatting outside of the SQL query.
Return only the SQL query.
""",
    "user_prompt": """Using the following database schema context, generate a single valid T-SQL query in response to the user's question.
        
Schema:
{schema_context}

User Question:
{query}

Important: Only return the SQL query. No additional explanation or comments.
"""
}

class SQLQueryGenerator(BaseTool):
    spec = ToolSpec(
        name="sql_query_generator",
        description="Generates an executable SQL query from a natural language question using the provided schema context.\n"
        "This tool should be used after identifying relevant tables and retrieving their schema or summaries.\n"
        "It is capable of reasoning over:\n"
        "  - Table relationships (joins)\n"
        "  - Filter conditions (e.g., status = 'active')\n"
        "  - Aggregations, groupings, and orderings\n"
        "  - Nested queries or UNION ALL if needed\n"
        "IMPORTANT:\n"
        "- Always refine the input query to produce correct SQL.\n"
        "- If previous attempts failed due to missing columns or filters, the agent must revise the input query accordingly.\n"
        "- Do NOT re-use the original user query blindly if it caused an error.\n"
        "- Inputs must be descriptive and specific to produce correct SQL.",
        when_to_use="Use this tool to generate an executable SQL query from a natural language question using the provided schema context.",
        inputs=[
            ToolParam(name="query", type="string", description="A refined natural language question that clearly reflects what the SQL should do.", required=True),
        ],
        returns=ToolReturn(type="string", description="A syntactically valid SQL query that can be executed on the provided schema."),
    )
    register: bool = True

    def __init__(
        self,
        llm: BaseModel = None,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.llm = llm
        self.logger = logger
        self.prompt = SQL_QUERY_GENERATION_PROMPT
        self.max_new_tokens = 2_000
        self.temperature = 0.7
        self.special_token: str = " [__SQL_QUERY__] "

    def __call__(self, query: str, **kwargs: Any) -> ToolResponse:
        schema_context = kwargs.get("schema_context", "")
        system_prompt = self.prompt["system_prompt"]
        user_prompt = self.prompt["user_prompt"].format(schema_context=schema_context, query=query)
        response = self.llm.ask(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            stream=False
        ).choices[0].message.content

        sql_query = response.replace("```sql", "").replace("`", "").strip()
        return ToolResponse(final_answer=False, observation=sql_query, extras={"sql_query": sql_query})
