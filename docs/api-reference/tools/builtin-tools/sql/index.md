# SQL Tools

A suite of **composable tools** for text‑to‑SQL workflows and database insights. Use them end‑to‑end or pick individual blocks that fit your agent.

<div class="grid cards" markdown>

-   :material-table-search:{ .lg .middle } **Relevant Table Schema Finder (LLM)**

    ---

    Selects the most relevant tables based on summaries and returns a schema context to guide query generation.

    [:octicons-arrow-right-24: Documentation](./relevant_table_schema_retriever.md)

-   :material-file-code-outline:{ .lg .middle } **SQL Query Generator**

    ---

    Produces a single, syntactically valid T‑SQL query from a refined question and provided schema context.

    [:octicons-arrow-right-24: Documentation](./sql_query_generator.md)

-   :material-database-arrow-right:{ .lg .middle } **SQL Executor**

    ---

    Executes the generated SQL with SQLAlchemy and returns rows as dictionaries for downstream formatting.

    [:octicons-arrow-right-24: Documentation](./sql_executor.md)

-   :material-table-large:{ .lg .middle } **Final Answer Formatter**

    ---

    Converts raw rows into a clear, user‑friendly narrative and/or markdown table suitable for UI display.

    [:octicons-arrow-right-24: Documentation](./final_answer_formatter.md)

-   :material-database-eye:{ .lg .middle } **DB Insights Tool**

    ---

    Answers conceptual questions about schema design and relationships using table metadata (no query execution).

    [:octicons-arrow-right-24: Documentation](./db_insights_tool.md)

</div>

---

## Flow (typical)

1. **Relevant Table Schema Finder (LLM)** → build focused schema context  
2. **SQL Query Generator** → produce a valid query  
3. **SQL Executor** → run the query and collect rows  
4. **Final Answer Formatter** → present the result to the user  
(Use **DB Insights Tool** anytime you need architectural/relationship explanations.)

> All SQL tools are standard `BaseTool` subclasses. Register them in a `Toolkit`, validate inputs with `ToolSpec.check_inputs()`, and chain state via `ToolResponse.extras` (e.g., `schema_context`, `db_results`).