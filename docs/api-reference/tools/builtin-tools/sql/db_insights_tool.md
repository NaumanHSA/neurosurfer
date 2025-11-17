# Database Insights Tool

> Module: `neurosurfer.tools.sql.db_insights_tool.DBInsightsTool`  
Pairs with: [`BaseTool`](../../base-tool.md) • [`ToolSpec`](../../tool-spec.md) • [`Toolkit`](../../toolkit.md)

## Overview
`DBInsightsTool` answers **high-level, conceptual questions** about your database using table **summaries/metadata** (not by executing SQL). It’s ideal for explaining the **purpose of the database**, the **roles of tables**, and **relationships** that may have architectural or security implications.

It builds a prompt from your schema store and queries your LLM to produce a **structured**, **concise** narrative answer.

---

## When to Use
Use this tool when you need:
- Conceptual insights about the **overall database design**.
- Explanations about how **entities relate** (e.g., departments ↔ roles ↔ access control).
- Architectural/security results that **cannot** be answered by running a SQL query.

Not for: returning data rows or executing queries (use [`SQLExecutor`](./sql_executor.md) for that).

---

## Spec (Inputs & Returns)

| Field | Type | Required | Description |
|---|---|:---:|---|
| `query` | `str` | ✓ | A natural-language question about the database's structure, design, or semantics. |

**Returns:** `str` — A natural-language explanation synthesizing table metadata and relationships.

> Note: `ToolSpec` here declares types as `str` (alias of `string` supported by the validator).

---

## Runtime Dependencies & Config

- **Constructor:** `DBInsightsTool(llm: BaseChatModel, sql_schema_store: SQLSchemaStore, logger: logging.Logger | None = None)`
- **Uses:** `sql_schema_store.store: Dict[str, {{summary: str, ...}}]`
- **Prompt:** Built from `DATABASE_INSIGHT_PROMPT`:
  - `system_prompt`: senior DB architect persona
  - `user_prompt`: injects your `query` and aggregated **table summaries**
- **LLM call:** `llm.ask(system_prompt, user_prompt, temperature=0.7, max_new_tokens=3000, stream=True)`
- **Streaming:** `True` by default → `results` may be a **generator**.
- **`final_answer`:** Defaults to `True` (can be overridden via `kwargs["final_answer"]`).

---

## Behavior

1. Collects table summaries via `get_tables_summaries__()`.
2. Renders **system** and **user** prompts.
3. Calls `llm.ask(...)` and returns a `ToolResponse` with `final_answer=True` by default (streaming enabled).

**Special token:** `" [__DATABASE_INSIGHT__] "` is available on the instance if you need to tag content for downstream parsing.

---

## Usage

```python
tool = DBInsightsTool(llm=chat_llm, sql_schema_store=schema_store)
resp = tool(query="How do departments relate to approvals?")
# resp.results may be a streaming generator (depending on llm.ask)
```

---

## Error Handling & Notes

- If the schema store lacks summaries, answers may be limited; the prompt **asks** the LLM to say “don’t know” where appropriate.
- This tool does **not** execute SQL and does not inspect live data.
- Ensure `sql_schema_store.store` contains `{{table_name: {{'summary': '...'}}}}` for best results.