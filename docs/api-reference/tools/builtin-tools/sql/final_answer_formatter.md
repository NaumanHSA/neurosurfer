# Final Answer Formatter

> Module: `neurosurfer.tools.sql.final_answer_formatter.FinalAnswerFormatter`  
Pairs with: [`BaseTool`](../../base-tool.md) • [`ToolSpec`](../../tool-spec.md) • [`Toolkit`](../../toolkit.md)

## Overview
`FinalAnswerFormatter` turns **raw SQL results** into a **user-friendly** explanation or **markdown table**, suitable as the **final answer** to present to end users. It leverages an LLM to produce clear, concise output while avoiding unnecessary SQL jargon.

---

## When to Use
- After [`SQLExecutor`](./sql_executor.md) has run a query and produced `db_results`.  
- You want a **polished** response (narrative, table, or mixed) for the **final** user-facing message.

Not for: generating or executing SQL. For generation, see [`SQLQueryGenerator`](./sql_query_generator.md).

---

## Spec (Inputs & Returns)

| Field | Type | Required | Description |
|---|---|:---:|---|
| `user_query` | `string` | ✓ | The original user question (for context). |

**Requires (via `kwargs`):** `db_results: list[dict]` — typically passed from `SQLExecutor` as `extras["db_results"]`.

**Returns:** `string` — Natural language summary and/or markdown table.

---

## Runtime Dependencies & Config

- **Constructor:** `FinalAnswerFormatter(llm: BaseModel, logger: logging.Logger | None = None)`
- **Prompt:** `RESULTS_PRESENTATION_PROMPT` (system + user)
- **Streaming:** `True`
- **Temperature:** `0.7`
- **Max tokens:** `8000`

**Preprocessing:** First 10 rows are formatted into a markdown table for the LLM (headers inferred from the first row’s keys).

---

## Behavior

1. Validates `db_results` from `kwargs`; if missing/empty, returns `ToolResponse(final_answer=False, results="No results found...")`.
2. Builds a concise table preview (up to 10 rows).
3. Calls `llm.ask(...)` to produce the final explanation/table and returns `ToolResponse(final_answer=True, results=...)`.

---

## Usage

```python
# After SQLExecutor
fmt = FinalAnswerFormatter(llm=chat_llm)
resp = fmt(user_query="Top 5 active users by posts?", db_results=db_results)
# resp.final_answer == True; resp.results is the final message (may stream)
```

---

## Error Handling & Notes

- If `db_results` is empty or missing, the tool returns a non-final results so the agent can branch (e.g., ask for a refined query).
- The tool streams by default; ensure your agent can stream generators from `ToolResponse.results`.