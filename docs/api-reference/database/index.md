# Databases

Neurosurfer’s **database module** provides production-grade utilities for connecting to data backends, inspecting schemas, and powering SQL agents. Today the focus is SQL via SQLAlchemy; the API is designed so additional backends (e.g., NoSQL) can be added without disrupting callers.

<div class="grid cards" markdown>

-   :material-database-cog:{ .lg .middle } **SQLDatabase**

    ---

    High-level SQLAlchemy wrapper with schema introspection, sample rows, view support, table filters, and **metadata caching** for fast startup.

    [:octicons-arrow-right-24: Documentation](./sql_database.md)

-   :material-file-document:{ .lg .middle } **SQLSchemaStore**

    ---

    Builds a compact, LLM-friendly schema knowledge base from a live database. Stores JSON summaries per table (optionally LLM-generated) for downstream tools/agents.

    [:octicons-arrow-right-24: Documentation](./sql_schema_store.md)

-   :material-database-plus:{ .lg .middle } **Future Backends**

    ---

    The public contracts are backend-agnostic. More database types (e.g., document stores) can be added under `neurosurfer.db.*` as the project grows.

    [:octicons-arrow-right-24: Contribute](../../contributing.md)

</div>

---

## How the SQL path works

1. **Connect** with `SQLDatabase` → reflect schema (tables/views), cache metadata, and (optionally) sample rows for context.  
2. **Summarize** with `SQLSchemaStore` → persist compact JSON summaries of each table’s purpose + raw schema (optionally via LLM).  
3. **Use in tools/agents** → pass stored summaries/schema to built-in SQL tools (e.g., relevant-table finder, query generator, executor) to enable a robust text-to-SQL workflow.

> The **schema cache** prevents repetitive network calls on every run. You control freshness via TTL and `force_refresh`.

---

## Quick Start

```python
from neurosurfer.db.sql.sql_database import SQLDatabase
from neurosurfer.db.sql.sql_schema_store import SQLSchemaStore
from neurosurfer.models.chat_models.openai_like import OpenAILikeModel  # example interface

# 1) Inspect a live database
db = SQLDatabase(
    database_uri="postgresql://user:pass@localhost:5432/analytics",
    include_tables=["users", "orders", "order_items"],  # optional
    sample_rows_in_table_info=3,
    view_support=True,
    metadata_cache_dir="~/.cache/neurosurfer/sqlmeta",
    cache_ttl_seconds=86400  # 1 day
)
print(db.get_table_info(["users"]))   # CREATE TABLE + sample rows

# 2) Build a schema store (with or without LLM summaries)
llm = OpenAILikeModel(model="gpt-4o-mini")  # any BaseChatModel-compatible LLM
store = SQLSchemaStore(db_uri="postgresql://user:pass@localhost:5432/analytics", llm=llm, storage_path="./")
for _ in store.train(summarize=True, force=True):  # generator yields progress counts
    pass

# 3) Consume summaries in tools/agents
print("Tables summarized:", store.get_tables_count())
users = store.get_table_data("users")
```

---

## Design goals

- **Frictionless introspection** – sensible defaults, easy filters, sample rows for realism.  
- **Repeatable performance** – schema caching avoids re-reflection every run.  
- **Agent-first** – JSON summaries that LLMs can reason over without expansive context windows.  
- **Safety** – no credentials in cache filenames; defensive error handling; opt-in view reflection.
