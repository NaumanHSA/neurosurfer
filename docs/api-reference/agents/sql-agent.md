# SQLAgent

**Module:** `neurosurfer.agents.sql_agent`

## Overview

`SQLAgent` specialises the [`ReActAgent`](react-agent.md) for relational databases. It boots with a SQL-focused [Toolkit](../tools/toolkit.md) that knows how to:

1. Summarise table schemas and cache them locally (`SQLSchemaStore`)
2. Pick the most relevant tables for a question
3. Generate a SQL statement
4. Execute the statement safely via SQLAlchemy
5. Format the result into natural language

Every tool call and observation is streamed to the caller, just like the base ReAct agent.

## Constructor

### `SQLAgent.__init__`

```python
SQLAgent(
    llm: BaseModel,
    db_uri: str,
    *,
    storage_path: str | None = None,
    sample_rows_in_table_info: int = 3,
    logger: logging.Logger = logging.getLogger(),
    verbose: bool = True,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `llm` | [`BaseModel`](../models/chat-models/base-model.md) | Model used for reasoning and SQL generation. |
| `db_uri` | `str` | SQLAlchemy connection string (e.g. `postgresql://user:pass@host/db`). |
| `storage_path` | `str \| None` | Optional location for persisting schema summaries. Defaults to the SQLSchemaStore default under the working directory. |
| `sample_rows_in_table_info` | `int` | Number of example rows to capture when summarising table schema. |
| `logger` | `logging.Logger` | Logger for status messages. |
| `verbose` | `bool` | When `True`, prints tool calls and observations as they happen. |

During initialisation the agent:

- Creates a SQLAlchemy engine via `create_engine(db_uri)`
- Instantiates `SQLSchemaStore` (handles schema discovery + caching)
- Registers the following tools in its toolkit:
  - `RelevantTableSchemaFinderLLM` *(table selection + schema fetch)*
  - `SQLQueryGenerator` *(LLM prompt -> SQL string)*
  - `SQLExecutor` *(read-only execution via the engine)*
  - `FinalAnswerFormatter` *(transforms result rows to natural language)*
  - `DBInsightsTool` *(high-level database summaries and health checks)*

You can register additional tools later via `register_tool`.

## Usage

```python
from neurosurfer.agents.sql_agent import SQLAgent
from neurosurfer.models.chat_models.openai import OpenAIModel

llm = OpenAIModel(model_name="gpt-4o-mini")

agent = SQLAgent(
    llm=llm,
    db_uri="sqlite:///examples/chinook.db",
    sample_rows_in_table_info=5,
    verbose=True,
)

chunks = []
for chunk in agent.run("List the top 5 artists by total sales."):
    print(chunk, end="")    # Stream thoughts + SQL outputs
    chunks.append(chunk)

transcript = "".join(chunks)
final_answer = transcript.split("<__final_answer__>")[-1].split("</__final_answer__>")[0].strip()
```

## Methods

### `run(user_query: str, **kwargs) -> Generator[str, None, str]`

Delegates to `ReActAgent.run_agent__`. Pass generation kwargs such as `temperature` or `max_new_tokens`. The generator yields formatted strings (thoughts, actions, tool observations) and finally returns the final answer string.

Runtime context (for example forcing a different temperature for a specific tool) can be supplied via keyword arguments; they are forwarded to every tool invocation.

### `register_tool(tool: BaseTool) -> None`

Adds a custom tool to the underlying toolkit and immediately updates the agent to include it in subsequent runs.

```python
agent.register_tool(MyAggregationTool(db_engine=agent.db_engine))
```

### `get_toolkit() -> Toolkit`

Helper used during initialisation. It exposes the preconfigured toolkit if you want to inspect or extend it.

### `train(summarize: bool = False, force: bool = False) -> Generator`

Convenience wrapper around `SQLSchemaStore.train`. Use it to pre-compute or refresh schema summaries outside of a chat session.

```python
for step in agent.train(summarize=True, force=False):
    print(step)
```

Available options depend on `SQLSchemaStore`; typically `summarize=True` asks the LLM for narrative summaries of each table, while `force=True` bypasses existing cache entries.

## Notes & best practices

- **Read-only by design**: `SQLExecutor` executes queries as-is. Ensure your database user has the appropriate permissions if you want to allow `UPDATE`/`INSERT`. For production systems we recommend read-only credentials.
- **Schema cache location**: supply `storage_path` if you need deterministic cache placement (for example when running in containers).
- **Streaming UI**: parse the generator output to differentiate between tool observations (prefixed with `[🔧]`/`Observation:`) and the final answer markers.
- **Additional context**: pass extra keyword arguments to `run(...)`—they will be merged into every tool call via `execute_llm_tool_output`. This is a convenient hook for injecting runtime config.
