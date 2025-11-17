import logging
from typing import Optional, List, Any, Generator

from ...models.chat_models.base import BaseChatModel
from ..base_tool import BaseTool, ToolResponse
from ..tool_spec import ToolSpec, ToolParam, ToolReturn


RESULTS_PRESENTATION_PROMPT = {
    "system_prompt": """You are a helpful assistant that explains SQL query results in a clear and user-friendly way.

Guidelines:
- Be adaptive: generate answers in plain natural language, tables or headings approach according to the need.
- Avoid SQL jargon unless explicitly asked.
- If no results are found, say so directly.
- Do not add unnecessary explanations or repeat the user's question.
- Focus on clarity, readability, and professionalism.
""",
    "user_prompt": """User Question:
{user_query}

Raw SQL Results:
{results}

Your Explanation:
"""
}

class FinalAnswerFormatter(BaseTool):
    spec = ToolSpec(
        name="final_answer_formatter",
        description="Use this tool to generate the FINAL ANSWER for the user **Only When you get raw SQL results by using sql_executor tool**.\n"
        "This tool converts raw SQL execution results into a clear, user-friendly explanation or markdown table "
        "using an LLM. It is essential for formatting responses that will be shown to end users in natural language.\n"
        "Use this tool when:\n"
        "- A SQL query has been executed.\n"
        "- You need to present the results clearly to the user.\n"
        "- You are about to respond with the final answer.\n",
        when_to_use="Use this tool to generate the FINAL ANSWER for the user **Only When you get raw SQL results by using sql_executor tool**.",
        inputs=[
            ToolParam(name="user_query", type="string", description="The original question asked by the user.", required=True),
        ],
        returns=ToolReturn(type="string", description="A natural language summary or markdown-formatted table suitable for UI display."),
    )

    def __init__(
        self,
        llm: BaseChatModel,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.llm = llm
        self.logger = logger
        self.prompt = RESULTS_PRESENTATION_PROMPT
        self.temperature = 0.7
        self.stream = True
        self.max_new_tokens = 8000

    def __call__(
        self,
        user_query: str,
        **kwargs: Any,
    ) -> ToolResponse:
        db_results = kwargs.get("db_results", [])

        if not db_results:
            return ToolResponse(final_answer=False, results=f"No results found for query: {user_query}")

        processed_results = self.preprocess_sql_results_for_llm(user_query=user_query, db_results=db_results)
        user_prompt = self.prompt["user_prompt"].format(
            user_query=user_query,
            results=processed_results
        )
        system_prompt = self.prompt["system_prompt"]
        response = self.llm.ask(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            stream=self.stream
        )
        return ToolResponse(final_answer=True, results=response)
    
    def preprocess_sql_results_for_llm(self, user_query: str, db_results: List[dict]) -> str:
        columns = list(db_results[0].keys())
        display_rows = db_results[:10]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = "\n".join(
            "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
            for row in display_rows
        )
        markdown_table = f"{header}\n{separator}\n{body}"
        return f"""User Query:\n{user_query}\n\nQuery Results (first {len(display_rows)} rows):\n{markdown_table}"""
