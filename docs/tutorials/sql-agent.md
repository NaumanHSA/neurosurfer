# SQL Agent

Give an agent read-only access to a database and let it answer questions in plain English —
then compare the two agent strategies, `ReactAgent` and `AgenticLoop`, on the same task.

## The `sql` tool

Neurosurfer ships a built-in `sql` tool (`neurosurfer.registry.core.database`). It speaks whatever
SQLAlchemy speaks — SQL Server, PostgreSQL, MySQL — and it **cannot modify data**: only
`SELECT` / `WITH` / `EXPLAIN` / `SHOW` are permitted.

It is one tool with four **operations**, which is the order you should work in:

| Operation | What it does |
|---|---|
| `test_connection` | Check the connection and report the server version. |
| `list_tables` | List tables as schema-qualified names. |
| `table_schema` | Columns with types and nullability, primary key, foreign keys. |
| `query` | Run one read-only query and return rows as a table. |

Doing it in that order is what makes a generated query correct. A table name guessed from the
request is the most common reason a generated query fails, and checking the connection first turns
a wrong password into one clear message instead of a failure part-way through a query.

## The connection string is a secret

The tool declares `dsn` as a **secret input**, so it never travels through a prompt. Store the
connection URL and reference it as `${NAME}`:

```
postgresql+psycopg://user:pw@host/db
mssql+pyodbc://user:pw@host:1433/db?driver=ODBC+Driver+18+for+SQL+Server
```

In a workflow the reference goes in `tool_args` on a node that declared it in `secrets:` — see
[State & secrets](../graph/index.md). The value reaches the tool and never the model.

## Two agents, one task

The interesting comparison is running the **same task through both agent types**:

- **`AgenticLoop`** — uses the provider's **native** tool-calling. Cleaner and more reliable when
  the model supports it.
- **`ReactAgent`** — drives the tool by **parsing text** (ReAct). Works with local models that lack
  a native tool API, at the cost of some robustness.

```python
# same tools + prompt, two agents
common = dict(
    provider=provider,
    tools=sql_pool,
    system_prompt="Answer using the sql tool. List tables and read the schema before querying, then finish.",
    guardrails=Guardrails(),
    io=AutoIO(),
    cwd=Path.cwd(),
)

loop = AgenticLoop(**common)
react = ReactAgent(**common)

for agent in (loop, react):
    result = await agent.run_collect("How many orders shipped last month?")
    print(type(agent).__name__, "→", result.final_text)
```

Turn on [observability](../observability/index.md) to see each agent's turns and tool calls side by
side in a trace — it is the fastest way to feel the difference between native tool-calling and
text-parsed ReAct.

## Next

- [Tools Catalog](../guides/tools-catalog.md) — every built-in tool, by domain.
- [Agents](../guides/agents.md) — the agent family and when to reach for each.
- [Graph & Workflows](../graph/index.md) — running the same thing as a repeatable workflow.

**Back to:** [Tutorials overview](index.md)
