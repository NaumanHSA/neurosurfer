"""
SQL Agent Module
================

Specialized ReAct agent for SQL databases built on top of the new ReActAgent core.

Workflow:
 1) Understand user's question
 2) Select relevant tables from cached schema summaries
 3) Retrieve detailed schema (if needed)
 4) Generate SQL
 5) Execute SQL safely
 6) Format results to natural language

Key Features:
 - Automatic schema discovery & caching (SQLSchemaStore)
 - Intelligent table selection
 - SQL generation with error recovery
 - Safe execution (via tool), no direct DB ops in LLM
 - Human-friendly final answer formatting
"""

from __future__ import annotations

from typing import Optional, Generator, Any
import logging
import sqlalchemy
from sqlalchemy import create_engine

from neurosurfer.agents.react import ReActAgent, ReActConfig
from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools import Toolkit
from neurosurfer.tools.base_tool import BaseTool
from neurosurfer.tools.sql import (
    RelevantTableSchemaFinderLLM,
    SQLExecutor,
    SQLQueryGenerator,
    FinalAnswerFormatter,
    DBInsightsTool,
)
from neurosurfer.db import SQLSchemaStore

# Extra nudges specific to SQL so the general ReAct loop stays on-rails
AGENT_SPECIFIC_INSTRUCTIONS = """
## SQL Agent Policy

- Never claim results without executing the SQL via the `sql_executor` tool.
- Prefer the following step plan unless strong evidence suggests otherwise:
  1) Use `relevant_table_schema_finder_llm` to pick tables and get concise schema context.
  2) Use `sql_query_generator` to craft a complete, dialect-correct query.
  3) Use `sql_executor` to run the query.
  4) Use `final_answer_formatter` to present the result in natural language.
- If a query fails:
  - For syntax errors: fix the SQL and retry.
  - For missing columns/tables: revisit `relevant_table_schema_finder_llm` with what you learned and regenerate.
- Do not set `"final_answer": true` for schema discovery or query generation tools.
- Only finalize after a successful `sql_executor` run (and optional formatting).
"""


class SQLAgent(ReActAgent):
    """
    SQL-aware ReActAgent with DB connection, schema cache, and SQL tools pre-wired.

    Example:
        >>> llm = TransformersModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
        >>> agent = SQLAgent(llm=llm, db_uri="sqlite:///my.db", sample_rows_in_table_info=3)
        >>> for chunk in agent.run("How many users registered last month?"):
        ...     print(chunk, end="")
    """

    def __init__(
        self,
        llm: BaseChatModel,
        db_uri: str,
        storage_path: Optional[str] = None,
        sample_rows_in_table_info: int = 3,
        logger: logging.Logger = logging.getLogger(__name__),
        verbose: bool = True,
        config: Optional[ReActConfig] = None,
        specific_instructions: Optional[str] = None,
    ) -> None:
        """
        Args:
            llm: Language model used by the agent.
            db_uri: SQLAlchemy-style URI (e.g., postgresql://user:pass@host/db).
            storage_path: Optional path to persist schema summaries.
            sample_rows_in_table_info: How many example rows to include in schema summaries.
            logger: Logger instance.
            verbose: Pass-through to ReActAgent (controls rich debug prints).
            config: ReAct configuration (retries, pruning, etc.). If None, defaults are used.
            specific_instructions: Extra system addendum. If None, SQL defaults are used.
        """
        self.llm = llm
        self.logger = logger
        self.verbose = verbose
        self.db_uri = db_uri

        # Connect DB and load schema store
        try:
            self.db_engine: sqlalchemy.Engine = create_engine(self.db_uri)
            self.sql_schema_store = SQLSchemaStore(
                db_uri=self.db_uri,
                llm=self.llm,
                sample_rows_in_table_info=sample_rows_in_table_info,
                storage_path=storage_path,
                logger=self.logger,
            )
            self.logger.info("[SQLAgent] Connected to database successfully.")
            self.logger.info(f"[SQLAgent] Loaded {len(self.sql_schema_store.store)} schema summaries.")
        except Exception as e:
            raise Exception(f"[SQLAgent] Failed to initialize DB or schema store: {e}")

        # Build the SQL toolkit
        self.toolkit = self._build_toolkit()

        # Initialize the parent ReActAgent with SQL-specific prompt addendum
        super().__init__(
            toolkit=self.toolkit,
            llm=self.llm,
            logger=self.logger,
            specific_instructions=specific_instructions or AGENT_SPECIFIC_INSTRUCTIONS,
            config=config,
        )

    # ---------- Public helpers ----------
    def train(self, summarize: bool = False, force: bool = False) -> Generator[str, None, None]:
        """
        Warm up schema cache (optionally summarize). This yields progress strings.
        """
        return self.sql_schema_store.train(summarize=summarize, force=force)

    def is_trained(self) -> bool:
        """True if we have at least one cached schema summary."""
        return len(self.sql_schema_store.store) > 0

    def register_tool(self, tool: BaseTool) -> None:
        """Register an extra tool and refresh the parent toolkit reference."""
        self.toolkit.register_tool(tool)

    # ---------- Internals ----------
    def _build_toolkit(self) -> Toolkit:
        tk = Toolkit()
        tk.register_tool(
            RelevantTableSchemaFinderLLM(
                llm=self.llm,
                sql_schema_store=self.sql_schema_store,
            )
        )
        tk.register_tool(SQLQueryGenerator(llm=self.llm))
        tk.register_tool(SQLExecutor(db_engine=self.db_engine))
        tk.register_tool(FinalAnswerFormatter(llm=self.llm))
        tk.register_tool(DBInsightsTool(llm=self.llm, sql_schema_store=self.sql_schema_store))
        return tk

    # Note: We do NOT override run(). We use ReActAgent.run() directly, which:
    # - streams LLM thoughts and final answers
    # - parses & repairs Actions
    # - executes tools with retries and input pruning
    # If needed, you can still call: for chunk in agent.run("..."): print(chunk, end="")
