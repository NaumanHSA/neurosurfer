# SQLAgent

**Module:** `neurosurfer.agents.sql_agent`

## Overview

`SQLAgent` specializes the [`ReActAgent`](./react-agent.md) for relational databases. It boots with a SQL-focused [`Toolkit`](../tools/toolkit.md) that knows how to:

1. Summarize table schemas and cache them locally (`SQLSchemaStore`)
2. Pick the most relevant tables for a question
3. Generate a SQL statement
4. Execute the statement safely via SQLAlchemy
5. Format the result into natural language

Every step is streamed like the base ReAct agent (thoughts → action → results → … → final). If your app suppresses special tokens, set `skip_special_tokens=True` in `ReActConfig` when constructing the agent (see below).

---

## Constructor

### `SQLAgent.__init__`

```python
SQLAgent(
    llm: BaseChatModel,
    db_uri: str,
    *,
    storage_path: str | None = None,
    sample_rows_in_table_info: int = 3,
    logger: logging.Logger = logging.getLogger(__name__),
    verbose: bool = True,
    config: ReActConfig | None = None,
    specific_instructions: str | None = None,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `llm` | [`BaseChatModel`](../models/chat-models/base-model.md) | Model used for reasoning and SQL generation. |
| `db_uri` | `str` | SQLAlchemy connection string (e.g. `postgresql://user:pass@host/db`). |
| `storage_path` | `str \| None` | Optional location for persisting schema summaries. Defaults to the `SQLSchemaStore` default under the working directory. |
| `sample_rows_in_table_info` | `int` | Number of example rows to capture when summarizing table schema. |
| `logger` | `logging.Logger` | Logger for status messages. |
| `verbose` | `bool` | When `True`, prints tool calls and results as they happen. |
| `config` | [`ReActConfig`](react-agent.md#reactconfig) \| `None` | Advanced configuration (retries, pruning, streaming markers, etc.). Defaults are used when `None`. |
| `specific_instructions` | `str \| None` | Optional SQL-specific system addendum. If `None`, sensible SQL policies are used (discover → generate → execute → format). |

During initialization the agent:

- Creates a SQLAlchemy engine via `create_engine(db_uri)`
- Instantiates `SQLSchemaStore` (handles schema discovery + caching)
- Registers the following tools in its toolkit:
  - [`RelevantTableSchemaFinderLLM`](../tools/builtin-tools/sql/relevant_table_schema_retriever.md) *(table selection + schema fetch)*
  - [`SQLQueryGenerator`](../tools/builtin-tools/sql/sql_query_generator.md) *(LLM prompt → SQL string)*
  - [`SQLExecutor`](../tools/builtin-tools/sql/sql_executor.md) *(read-only execution via the engine)*
  - [`FinalAnswerFormatter`](../tools/builtin-tools/sql/final_answer_formatter.md) *(transforms rows to natural language)*
  - [`DBInsightsTool`](../tools/builtin-tools/sql/db_insights_tool.md) *(high‑level database summaries and health checks)*

> **Note:** If your docs use a different path layout, adjust the links above to match your structure.

---

## Usage

```python
from neurosurfer.agents.sql_agent import SQLAgent
from neurosurfer.agents.react import ReActConfig
from neurosurfer.models.chat_models.openai import OpenAIModel

llm = OpenAIModel(model_name="gpt-4o-mini")

agent = SQLAgent(
    llm=llm,
    db_uri="sqlite:///examples/chinook.db",
    sample_rows_in_table_info=5,
    verbose=True,
    config=ReActConfig(skip_special_tokens=False, allow_input_pruning=True, repair_with_llm=True),
)

for chunk in agent.run("List the top 5 artists by total sales."):
    print(chunk, end="")  # Streams thoughts + tool results + final answer markers
```

If you set `skip_special_tokens=True` in `ReActConfig`, the agent will **not** emit `<__final_answer__>` markers; only the raw text streams. This is useful if your UI has its own finalization logic.

---

## Methods

### `run(user_query: str, **kwargs) -> Generator[str, None, str]`

Delegates to `ReActAgent.run`. Pass generation kwargs such as `temperature` or `max_new_tokens`. The generator yields formatted strings (thoughts, actions, tool results) and finally returns the final answer string.

Runtime context (for example injecting a runtime filter) can be supplied via keyword arguments; they are forwarded to every tool invocation.

### `register_tool(tool: BaseTool) -> None`

Adds a custom tool to the underlying toolkit and immediately updates the agent to include it in subsequent runs.

```python
agent.register_tool(MyAggregationTool(db_engine=agent.db_engine))
```

### `get_toolkit() -> Toolkit`

Returns the preconfigured toolkit if you want to inspect or extend it.

### `train(summarize: bool = False, force: bool = False) -> Generator`

Convenience wrapper around `SQLSchemaStore.train`. Use it to pre-compute or refresh schema summaries outside of a chat session.

```python
for step in agent.train(summarize=True, force=False):
    print(step)
```
Available options depend on `SQLSchemaStore`; typically `summarize=True` asks the LLM for narrative summaries of each table, while `force=True` bypasses existing cache entries.

### `is_trained() -> bool`

Returns `True` when at least one cached schema summary is available.

---

## Built-in SQL tools

| Tool | Purpose | Typical Inputs | Typical Output |
| --- | --- | --- | --- |
| [`RelevantTableSchemaFinderLLM`](../tools/builtin-tools/sql/relevant_table_schema_retriever.md) | Selects relevant tables and returns concise schema context (optionally with sample rows). | `question: str`, optional knobs for limits | Schema text / JSON for downstream SQL generation |
| [`SQLQueryGenerator`](../tools/builtin-tools/sql/sql_query_generator.md) | Produces a dialect-correct SQL query from the question + schema context. | `question: str`, `schema: str` | SQL string |
| [`SQLExecutor`](../tools/builtin-tools/sql/sql_executor.md) | Executes a SQL statement via the configured SQLAlchemy engine. | `sql: str`, optional params | Rows (list of dicts/tuples) or error info |
| [`FinalAnswerFormatter`](../tools/builtin-tools/sql/final_answer_formatter.md) | Converts rows into a concise, human-readable answer. | `rows: Any`, `question: str` | Natural-language summary text |
| [`DBInsightsTool`](../tools/builtin-tools/sql/db_insights_tool.md) | Provides quick DB-wide insights (table counts, schema stats, anomalies). | optional scope params | Descriptive text / small tables |

> Each tool defines a `ToolSpec` so the ReAct agent can validate inputs, prune unknown keys (if enabled), and self-repair Actions when needed.

---

## Notes & Best Practices

- **Read-only by default:** `SQLExecutor` will execute whatever SQL you pass it. For production, use read-only credentials unless write operations are explicitly intended and guarded.  
- **Schema cache location:** supply `storage_path` if you need deterministic cache placement (e.g., containers, CI).  
- **Streaming UI:** distinguish tool results (e.g., `Observation:` lines) from the final answer markers (unless suppressed).  
- **Domain policy:** the agent ships with SQL-specific guidance (discover → generate → execute → format). You can override/extend via `specific_instructions`.  
- **Extras between tools:** tool `extras` are passed **agent-side** to the next tool call without going through the LLM. Use this to pass rich Python objects (e.g., compiled queries, parsed schemas) that aren’t easily serializable.  
- **Retries & repair:** Action parsing problems and tool failures are automatically repaired with bounded retries (see [`ReActConfig`](react-agent.md#reactconfig) and [`RetryPolicy`](react-agent.md#retrypolicy)).

---

## Security Considerations

- **SQL injection:** The agent itself won’t construct queries unsafely if `SQLQueryGenerator` is given correct schema context, but **validate inputs** and consider parameterized SQL in your execution layer.  
- **Privileges:** Provide least-privilege DB credentials. Separate read and write roles where possible.  
- **Auditing:** Log tool calls and queries in production (subject to privacy constraints).

---
