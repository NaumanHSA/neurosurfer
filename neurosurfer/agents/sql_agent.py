"""
SQL Agent Module
================

This module provides a specialized ReAct agent for SQL database interactions.
The SQLAgent extends ReActAgent with SQL-specific tools for schema discovery,
query generation, execution, and result formatting.

The SQLAgent workflow:
    1. Understand user's question
    2. Find relevant tables using schema summaries
    3. Retrieve detailed schema for those tables
    4. Generate SQL query
    5. Execute query
    6. Format results into natural language

Key Features:
    - Automatic schema discovery and caching
    - Intelligent table selection
    - SQL query generation with error recovery
    - Safe query execution
    - Natural language result formatting
    - Database insights and statistics

Example:
    >>> from neurosurfer.agents import SQLAgent
    >>> from neurosurfer.models.chat_models import TransformersModel
    >>> 
    >>> llm = TransformersModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
    >>> agent = SQLAgent(
    ...     llm=llm,
    ...     db_uri="postgresql://user:pass@localhost/mydb",
    ...     sample_rows_in_table_info=3
    ... )
    >>> 
    >>> # Ask questions about the database
    >>> for chunk in agent.run("How many users registered last month?"):
    ...     print(chunk, end="")
"""
from typing import Optional, Generator, Any
import logging
import sqlalchemy
from sqlalchemy import create_engine
from ..models.chat_models.base import BaseModel
from ..tools.base_tool import BaseTool
from .react_agent import ReActAgent
from ..tools import Toolkit
from ..tools.sql import (
    RelevantTableSchemaFinderLLM,
    SQLExecutor,
    SQLQueryGenerator,
    FinalAnswerFormatter,
    DBInsightsTool
)
from ..db import SQLDatabase, SQLSchemaStore

AGENT_SPECIFIC_INSTRUCTIONS = None
# AGENT_SPECIFIC_INSTRUCTIONS = """
# ## Typical Agent Workflow
# Use this as a general plan when answering questions using tools:

# 1. **Retrieve relevant tables schema**
#    - Use `relevant_table_schema_finder_llm` to get schema information for the identified tables.

# 2. **Generate SQL query**
#    - Use `sql_query_generator` with the user's question and the schema to create a SQL query.

# 3. **Execute the SQL query**
#    - Use `sql_executor` to run the query.

#    - **Handle errors**
#      - If the query fails:
#        - For **syntax errors**, revise the query and try again.
#        - For **missing information**, re-check relevant tables or schema.

# 4. **Present the result**
#    - Use `final_answer_formatter` to convert raw SQL results into a human-readable answer.


# NOTES:
# - Always execute the SQL query using tool `sql_executor` before generating the final answer. Do not ask the user for confirmation.

# """

class SQLAgent(ReActAgent):
    """
    Specialized ReAct agent for SQL database interactions.
    
    This agent extends ReActAgent with SQL-specific capabilities, including
    schema discovery, query generation, and result formatting. It automatically
    sets up the necessary tools and manages database connections.
    
    The agent uses a multi-step reasoning process:
    1. Analyzes the user's question
    2. Identifies relevant tables
    3. Retrieves schema information
    4. Generates SQL queries
    5. Executes queries safely
    6. Formats results naturally
    
    Attributes:
        llm (BaseModel): Language model for reasoning and generation
        db_uri (str): Database connection URI
        db_engine (sqlalchemy.Engine): SQLAlchemy engine
        sql_schema_store (SQLSchemaStore): Schema cache and manager
        toolkit (Toolkit): SQL-specific tools
    
    Example:
        >>> agent = SQLAgent(
        ...     llm=llm,
        ...     db_uri="sqlite:///mydb.db",
        ...     sample_rows_in_table_info=3,
        ...     verbose=True
        ... )
        >>> 
        >>> # Natural language queries
        >>> for chunk in agent.run("Show me top 10 customers by revenue"):
        ...     print(chunk, end="")
        >>> 
        >>> # Complex analytics
        >>> for chunk in agent.run("What's the average order value by month?"):
        ...     print(chunk, end="")
    """
    def __init__(
        self,
        llm: BaseModel,
        db_uri: str,
        storage_path: Optional[str] = None,
        sample_rows_in_table_info: int = 3,
        logger: logging.Logger = logging.getLogger(),
        verbose: bool = True
    ):
        """
        Initialize the SQL agent.
        
        Args:
            llm (BaseModel): Language model for reasoning
            db_uri (str): Database connection URI (e.g., "postgresql://user:pass@host/db")
            storage_path (Optional[str]): Path to store schema cache. Default: None (auto)
            sample_rows_in_table_info (int): Number of sample rows to include in schema.
                Default: 3
            logger (logging.Logger): Logger instance. Default: root logger
            verbose (bool): Enable verbose output. Default: True
        
        Raises:
            Exception: If database connection fails
        
        Example:
            >>> agent = SQLAgent(
            ...     llm=my_llm,
            ...     db_uri="postgresql://localhost/mydb",
            ...     sample_rows_in_table_info=5
            ... )
        """
        self.llm = llm
        self.logger = logger
        self.verbose = verbose
        self.db_uri = db_uri
        try:
            self.db_engine: sqlalchemy.Engine = create_engine(self.db_uri)
            self.sql_schema_store = SQLSchemaStore(
                db_uri=self.db_uri,
                llm=self.llm,
                sample_rows_in_table_info=sample_rows_in_table_info,
                storage_path=storage_path,
                logger=self.logger
            )
            self.logger.info(f"[SQLDatabase] Connected to database successfully.")
            self.logger.info(f"[SQLSchemaStore] Loaded {len(self.sql_schema_store.store)} schema summaries.")
        except Exception as e:
            raise Exception(f"[SQLAgent] Failed to connect to database: {e}")

        self.toolkit = self.get_toolkit()
        super().__init__(
            toolkit=self.toolkit,
            llm=self.llm,
            logger=self.logger,
            verbose=self.verbose,
            specific_instructions=AGENT_SPECIFIC_INSTRUCTIONS
        )

    def get_toolkit(self) -> Toolkit:
        # register tools here
        toolkit = Toolkit()
        toolkit.register_tool(RelevantTableSchemaFinderLLM(llm=self.llm, sql_schema_store=self.sql_schema_store))
        toolkit.register_tool(SQLExecutor(db_engine=self.db_engine))
        toolkit.register_tool(SQLQueryGenerator(llm=self.llm))
        toolkit.register_tool(FinalAnswerFormatter(llm=self.llm))
        toolkit.register_tool(DBInsightsTool(llm=self.llm, sql_schema_store=self.sql_schema_store))
        return toolkit

    def register_tool(self, tool: BaseTool):
        self.toolkit.register_tool(tool)
        self.update_toolkit(self.toolkit)

    def train(self, summarize: bool = False, force: bool = False) -> Generator:
        return self.sql_schema_store.train(summarize=summarize, force=force)

    def run(self, user_query: str, **kwargs: Any) -> Generator:
        return self.run_agent__(user_query, **kwargs)

    def is_trained(self) -> bool:
        return len(self.sql_schema_store.store) > 0
