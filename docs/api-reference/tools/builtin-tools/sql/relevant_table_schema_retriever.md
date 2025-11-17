# Relevant Table Schema Finder (LLM)

> Module: `neurosurfer.tools.sql.relevant_table_schema_retriever.RelevantTableSchemaFinderLLM`  
Pairs with: [`BaseTool`](../../base-tool.md) • [`ToolSpec`](../../tool-spec.md) • [`Toolkit`](../../toolkit.md)

## Overview
`RelevantTableSchemaFinderLLM` uses an LLM to **select the most relevant tables** for a user’s question based on **table summaries**, then retrieves the corresponding **schemas** from your schema store. It’s typically used **early** in a workflow before generating SQL.

---

## When to Use
- You have many tables and need to **narrow down** which are relevant to the question.
- You want to assemble a **schema context** to pass to [`SQLQueryGenerator`](./sql_query_generator.md).

---

## Spec (Inputs & Returns)

| Field | Type | Required | Description |
|---|---|:---:|---|
| `query` | `string` | ✓ | Natural-language user question. |

**Returns:** `string` — A human-readable message listing selected tables.  
**Extras:** `schema_context: string` — The formatted schemas of selected tables for downstream tools.

---

## Runtime Dependencies & Config

- **Constructor:** `RelevantTableSchemaFinderLLM(llm: BaseChatModel, sql_schema_store: SQLSchemaStore, logger: logging.Logger | None = None)`
- **Prompt:** `RELEVENT_TABLES_PROMPT` (note: result must be a valid Python list literal)
- **Top K:** `top_k = 6` (upper bound; LLM can return fewer)
- **Token trimming:** uses `RAGRetrieverAgent._trim_context_by_token_limit(...)` to fit summaries and adjust `max_new_tokens`.
- **LLM call:** `stream=False` (expects a one-shot list literal)
- **Post-processing:** `eval(...)` is applied to parse the returned list of table names.

> ⚠️ **Security note:** Since `eval` is used on the LLM response, ensure your LLM and prompts are trusted/controlled. The system prompt strictly instructs a **Python list literal only**—no text or newlines.

---

## Behavior

1. Collects table summaries: `"Table: <name>\nSummary: <summary>\n\n"`.
2. Trims context to token budget if needed.
3. Calls LLM; expects output like `['Users', 'Orders']`.
4. Builds a message and **schema context** by fetching schemas via `sql_schema_store.get_table_data(name)`.
5. Returns `ToolResponse(final_answer=False, results=<message>, extras={{"schema_context": ...}})`.

---

## Usage

```python
finder = RelevantTableSchemaFinderLLM(llm=chat_llm, sql_schema_store=schema_store)
resp = finder(query="Show monthly revenue by region in 2024")
schema_ctx = resp.extras["schema_context"]  # pass to SQLQueryGenerator
```

---

## Error Handling & Notes

- If the LLM returns an invalid list, `eval` may fail; guard at the agent layer if necessary.
- Ensure `sql_schema_store` contains both `summary` and `schema` for each table used.
- Special token available: `" [__RELEVANT_TABLES__] "` (if you tag content downstream).