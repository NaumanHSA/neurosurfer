import logging
from typing import Optional, List, Any

from neurosurfer.models.embedders.base import BaseEmbedder  
from neurosurfer.models.chat_models.base import BaseModel
from neurosurfer.db.sql_schema_store import SQLSchemaStore
from neurosurfer.agents.rag import RAGAgent

from ..base_tool import BaseTool, ToolResponse
from ..tool_spec import ToolSpec, ToolParam, ToolReturn


RELEVENT_TABLES_PROMPT = {
    "system_prompt": """You are a helpful assistant that selects the most relevant database tables for answering a user's query.
You are given documentation containing multiple table names and their summaries.
Based on the user query, return a valid Python list of the most relevant table names — up to {top_k}, but fewer if appropriate.
Prioritize precision and only include tables that clearly relate to the query.
Strictly return only a Python list of strings like ['TableA', 'TableB'] — no explanations, no extra text, no newlines.
This list will be parsed with eval(), so the format must be correct.
""",
    "user_prompt": """Given the following table summaries:

{tables_and_summaries}

Which tables are relevant to answer the following question:

"{user_query}"

Return ONLY a valid Python list of the most relevant table names — up to {top_k}, but fewer if appropriate.
"""
}


class RelevantTableSchemaFinderLLM(BaseTool):
    spec = ToolSpec(
        name="relevant_table_schema_finder_llm",
        description="This tool uses an LLM to analyze table schema and determine which tables are most relevant to a user's question."
        "Then it retrieves the schema of the relevant tables from the vectorstore. This tool is typically used early in an agent workflow to extract the schema structure,"
        "which can then be passed to other tools like SQLQueryGenerator.",
        when_to_use="Use this tool to retrieve the schema of the relevant tables from the vectorstore. This tool is typically used early in an agent workflow to extract the schema structure,"
        "which can then be passed to other tools like SQLQueryGenerator.",
        inputs=[
            ToolParam(name="query", type="string", description="Query in natural language for the tool which best describes the user's question.", required=True),
        ],
        returns=ToolReturn(type="string", description="A string containing the schema of the relevant tables."),
    )

    def __init__(
        self,
        llm: BaseModel,
        sql_schema_store: SQLSchemaStore,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.llm = llm
        self.sql_schema_store = sql_schema_store
        self.rag_agent = RAGAgent(llm=llm)
        self.logger = logger
        self.prompt = RELEVENT_TABLES_PROMPT
        self.top_k: int = 6
        self.max_new_tokens: int = 4096
        self.temperature: float = 0.3
        self.special_token: str = " [__RELEVANT_TABLES__] "

    def __call__(
        self,
        query: str,
        **kwargs: Any,
    ) -> ToolResponse:
        tables_and_summaries = self.get_tables_summaries__()
        system_prompt = self.prompt["system_prompt"].format(top_k=self.top_k)
        base_user_prompt = self.prompt["user_prompt"].format(user_query=query, tables_and_summaries="", top_k=self.top_k)
        # Now trim context tokens if needed
        trim_results = self.rag_agent._trim_context_by_token_limit(
            system_prompt=system_prompt,
            user_prompt=base_user_prompt,
            db_context=tables_and_summaries,
        )
        user_prompt = self.prompt["user_prompt"].format(
            user_query=query,
            tables_and_summaries=trim_results.trimmed_context,
            top_k=self.top_k
        )
        response = self.llm.ask(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_new_tokens=trim_results.final_max_new_tokens,
            stream=False
        )
        relevant_tables = eval(response["choices"][0]["message"]["content"]) if response else []
        message = "I have retrieved schema for the following tables which are relevant to answer the user query:\n" + ", ".join(relevant_tables)
        schema_context = message + "\n" + self.get_table_schema(relevant_tables)
        return ToolResponse(final_answer=False, observation=message, extras={"schema_context": schema_context})

    def get_tables_summaries__(self):
        table_summaries = ""
        for table_name, table_data in self.sql_schema_store.store.items():
            summary = table_data["summary"]
            table_info = f"""Table: {table_name}\nSummary: {summary}"""
            table_summaries += f"{table_info}\n\n"
        return table_summaries

    def get_table_schema(self, tables: List[str]) -> str:
        tables_schema = ""
        for table_name in tables:
            if table_name in self.sql_schema_store.store:
                table_data = self.sql_schema_store.get_table_data(table_name)
                tables_schema += f"\n\nTable Name: {table_name}\nSummary: {table_data['summary']}\n\nSchema: {table_data['schema']}\n"
                tables_schema += f"######################## {table_name} TABLE END ########################\n\n"    
        return tables_schema