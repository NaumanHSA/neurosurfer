# SQLDatabase

> Module: `neurosurfer.db.sql.sql_database.SQLDatabase`

A production-grade **SQLAlchemy** wrapper used by Neurosurfer’s SQL agents/tools. It provides **schema introspection**, **table/view filtering**, **sample rows**, and **metadata caching** so agents can plan reliably without repeated, slow reflection calls.

---

## Features

- 🔌 **SQLAlchemy-powered**: works with Postgres, MySQL, SQLite, SQL Server (via ODBC), etc.  
- 🧭 **Table/View discovery**: include/ignore lists + optional view reflection.  
- 🔎 **Schema text**: emits `CREATE TABLE` DDL plus optional indexes and sample rows.  
- 🚀 **Metadata cache**: reflective metadata is cached to disk; TTL and force refresh supported.  
- 🧪 **Safe query execution**: convenience `run(...)` pattern (if present in your codebase) and low-level helpers demonstrated below.  
- 🧰 **Utility**: `build_connection_string(...)` for consistent DSNs.

---

## Initialization

```python
from neurosurfer.db.sql.sql_database import SQLDatabase

db = SQLDatabase(
    database_uri="postgresql://user:pass@localhost:5432/analytics",
    schema=None,                         # target schema (optional)
    ignore_tables=None,                  # e.g., ["migrations"]
    include_tables=None,                 # OR a whitelist, e.g., ["users", "orders"]
    sample_rows_in_table_info=3,         # rows per table to include under /* ... */
    indexes_in_table_info=False,         # include index info in DDL block
    custom_table_info=None,              # {table_name: "pre-rendered info string"}
    view_support=False,                  # include views in reflection
    max_string_length=300,               # formatting clamp for sample values
    lazy_table_reflection=False,         # when False, reflect eagerly
    metadata_cache_dir="~/.cache/neurosurfer/sqlmeta",
    force_refresh=False,
    cache_ttl_seconds=86400,             # expire cache after N seconds (None = never)
)
```

### Parameter reference

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `database_uri` | `str | URL` | — | SQLAlchemy connection string. |
| `schema` | `str?` | `None` | Schema namespace to inspect (dialect-dependent). |
| `ignore_tables` | `list[str]?` | `None` | Exclude these tables from all operations. |
| `include_tables` | `list[str]?` | `None` | Restrict operations to these tables only. Mutually exclusive with `ignore_tables`. |
| `sample_rows_in_table_info` | `int` | `3` | Number of example rows to include per table in `get_table_info()`. |
| `indexes_in_table_info` | `bool` | `False` | Include index metadata in `get_table_info()` output. |
| `custom_table_info` | `dict?` | `None` | Pre-rendered table strings; bypass reflection for those tables. |
| `view_support` | `bool` | `False` | Reflect views in addition to tables. |
| `max_string_length` | `int` | `300` | Clamp cell values when printing sample rows. |
| `lazy_table_reflection` | `bool` | `False` | If `False`, reflect eagerly (or load from cache). If `True`, reflect on first use. |
| `metadata_cache_dir` | `str | Path?` | `~/.cache/sqlalchemy_metadata` | Where to store metadata cache files. |
| `force_refresh` | `bool` | `False` | Ignore cache and rebuild metadata now. |
| `cache_ttl_seconds` | `int?` | `None` | If set, reload metadata when cache age exceeds TTL. |

---

## Common tasks

### 1) List usable tables
```python
db.get_usable_table_names()
# honors include/ignore and (optional) view_support
```

### 2) Get schema info (DDL + extras)
```python
print(db.get_table_info())          # all usable tables
print(db.get_table_info(["users"])) # specific tables
```

Output contains `CREATE TABLE ...` followed by an optional `/* ... */` block with indexes and `N` sample rows, e.g.:

```
CREATE TABLE users (
  id BIGINT PRIMARY KEY,
  email TEXT NOT NULL,
  ...
)

/*
Table Indexes:
...

3 rows from users table:
id	email	...
1	alice@example.com	...
2	bob@example.com	...
3	...
*/
```

### 3) Build DSN safely
```python
from neurosurfer.db.sql.sql_database import SQLDatabase

dsn = SQLDatabase.build_connection_string(
    server="db.acme.com",
    database="analytics",
    username="svc_app",
    password="s3cr3t!",
    driver="mssql+pyodbc",
    odbc_driver="ODBC Driver 18 for SQL Server",
    port="1433",
)
```

---

## Metadata caching

When `lazy_table_reflection=False`, SQLDatabase attempts to load previously reflected `MetaData` from a cache file whose name is derived from:

- **dialect** (e.g., `postgresql`),  
- **database identifier** (never includes credentials),  
- **schema**, **view_support**, and the **set of usable tables**.

You control freshness and behavior with:

- `cache_ttl_seconds` — consider cache stale after this many seconds,  
- `force_refresh=True` — ignore cache and rebuild now.

> For `sqlite:///:memory:` the cache is skipped entirely.

---

## Notes & gotchas

- If both `include_tables` and `ignore_tables` are set → a `ValueError` is raised.  
- When asking for `table_names` that don’t exist → you’ll get a `ValueError`.  
- Sample rows use `SELECT ... LIMIT N`; some dialects may return `ProgrammingError` for empty tables (handled).  
- JSON/unknown-typed columns (`NullType`) are excluded from printed DDL for clarity.
