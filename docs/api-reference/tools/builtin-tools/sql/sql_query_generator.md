# SQL Query Generator

> Module: `neurosurfer.tools.sql.sql_query_generator.SQLQueryGenerator`  
Pairs with: [`BaseTool`](../../base-tool.md) • [`ToolSpec`](../../tool-spec.md) • [`Toolkit`](../../toolkit.md)

## Overview
`SQLQueryGenerator` uses an LLM to produce a **syntactically valid T‑SQL query** from a refined natural-language request and a **schema context**. It’s designed to incorporate **joins**, **filters**, and **aggregations**, and to avoid dangerous or ambiguous SQL patterns.

---

## When to Use
- After you’ve identified relevant tables and assembled schema context (e.g., using [`RelevantTableSchemaFinderLLM`](./relevant_table_schema_retriever.md)).
- When you need an executable query for [`SQLExecutor`](./sql_executor.md).

---

## Spec (Inputs & Returns)

| Field | Type | Required | Description |
|---|---|:---:|---|
| `query` | `string` | ✓ | A **refined** question describing exactly what the SQL should do. |

**Requires (via `kwargs`):** `schema_context: string` — The schema text for relevant tables.

**Returns:** `string` — A single T‑SQL query string (`extras["sql_query"]` mirrors it).

---

## Runtime Dependencies & Config

- **Constructor:** `SQLQueryGenerator(llm: BaseModel | None = None, logger: logging.Logger | None = None)`
- **Prompt:** `SQL_QUERY_GENERATION_PROMPT` (strict rules)
  - Avoid `*`; prefer `TOP n`; `LIKE '%value%'` for text matches; `=` for numeric/date
  - BIT columns: `1/0` for TRUE/FALSE
  - **Output only the query** — no explanations
- **LLM call:** `stream=False`, `max_new_tokens=2000`, `temperature=0.7`
- **Post-process:** removes code fences/backticks.

**Special token:** `" [__SQL_QUERY__] "` if you need to tag content.

---

## Behavior

1. Renders system & user prompts using the given `schema_context` and `query`.
2. Calls the LLM and extracts `response["choices"][0]["message"]["content"]`.
3. Strips code fences, returns `ToolResponse(final_answer=False, observation=sql_query, extras={{"sql_query": sql_query}})`.

---

## Usage

```python
gen = SQLQueryGenerator(llm=chat_llm)
sql = gen(query="Top 10 customers by total 2024 spend", schema_context=schema_ctx).extras["sql_query"]
```

---

## Best Practices & Safety

- **Refine the question** before calling this tool; vague prompts produce vague SQL.
- If a previous attempt failed (missing columns, invalid filters), **revise the input** and try again (the tool’s prompt explicitly instructs this behavior).
- Keep generation **read-only**; validate queries before execution in production.