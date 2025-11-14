import logging
from typing import Optional, Dict, Any

import sqlalchemy
from ..base_tool import BaseTool, ToolResponse
from ..tool_spec import ToolSpec, ToolParam, ToolReturn

class SQLExecutor(BaseTool):
    spec = ToolSpec(
        name="sql_executor",
        description="Executes a raw SQL query string using a provided SQLAlchemy database engine. "
        "This tool is typically used after a SQL query has been generated, allowing the agent to retrieve live results from the database.",
        when_to_use="Use this tool to execute a raw SQL query string using a provided SQLAlchemy database engine. "
        "This tool is typically used after a SQL query has been generated, allowing the agent to retrieve live results from the database.",
        inputs=[
            ToolParam(name="sql_query", type="string", description="The SQL query to execute.", required=True),
        ],
        returns=ToolReturn(type="list", description="A list of dictionaries, where each dictionary represents one row from the result set."),
    )

    def __init__(
        self,
        db_engine: sqlalchemy.Engine,
        logger: Optional[logging.Logger] = logging.getLogger(__name__),
    ):
        self.logger = logger
        self.special_token: str = " [__SQL_EXECUTOR__] "
        self.db_engine = db_engine

    def __call__(self, sql_query: str, **kwargs: Any) -> ToolResponse:
        try:
            # sql_query = kwargs.get("sql_query", "")
            db_results = self.get_results_with_columns(sql_query)
            if db_results:
                results = "I've executed the SQL query and have fetched the results."
            else:
                results = "I've executed the SQL query but no results were found."

            return ToolResponse(final_answer=False, results=results, extras={"db_results": db_results})
        except Exception as e:
            self.logger.error(f"SQL execution failed: {str(e)}")
            return ToolResponse(final_answer=False, results=[{"error": str(e)}])

    def get_results_with_columns(self, query: str):
        with self.db_engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(query))
            columns = result.keys()
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]