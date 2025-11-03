# SQL Executor

> Module: `neurosurfer.tools.sql.sql_executor.SQLExecutor`  
Pairs with: [`BaseTool`](../../base-tool.md) • [`ToolSpec`](../../tool-spec.md) • [`Toolkit`](../../toolkit.md)

## Overview
`SQLExecutor` runs a **raw SQL query string** using a provided **SQLAlchemy** engine and returns rows as a list of dictionaries. It is typically used **after** you’ve generated SQL via [`SQLQueryGenerator`](./sql_query_generator.md).

---

## When to Use
- You have a finalized SQL query and need to **execute** it against a live database.
- You want raw rows for further processing or for final formatting via [`FinalAnswerFormatter`](./final_answer_formatter.md).

---

## Spec (Inputs & Returns)

| Field | Type | Required | Description |
|---|---|:---:|---|
| `sql_query` | `string` | ✓ | The SQL to execute. |

**Returns:** `list` — Each item is a `dict` representing one result row.  
**Extras:** `db_results: list[dict]` — returned in `ToolResponse.extras` for downstream tools.

---

## Runtime Dependencies & Config

- **Constructor:** `SQLExecutor(db_engine: sqlalchemy.Engine, logger: logging.Logger | None = None)`
- **Executes:** `connection.execute(sqlalchemy.text(query))`
- **Marshalling:** zips `result.keys()` with each row to produce `dict` rows.
- **Observation message:** human-friendly (“fetched results” / “no results”).

**Special token:** `" [__SQL_EXECUTOR__] "` is available on the instance for tagging, if desired.

---

## Behavior

1. Executes the SQL using the configured `db_engine`.
2. Builds `db_results` (list of dict rows).
3. Returns `ToolResponse(final_answer=False, observation=<message>, extras={{"db_results": db_results}})`.

On exception: logs an error and returns `ToolResponse(final_answer=False, observation=[{{"error": "<message>"}}])`.

---

## Usage

```python
executor = SQLExecutor(db_engine=engine)
resp = executor(sql_query="SELECT TOP 5 id, name FROM users ORDER BY created_at DESC")
rows = resp.extras.get("db_results", [])
```

---

## Security & Notes

- Pass **parameterized** queries where possible to avoid injection (`sqlalchemy.text` with bindparams).
- Ensure the engine user has appropriate **read-only** permissions for analysis scenarios.