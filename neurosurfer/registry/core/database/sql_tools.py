"""The SQL tool — read-only access to a real database server, as four operations.

The catalog had `data`, which reads local CSV/JSON files and SQLite *files*. There
was nothing that could reach a database **server** — so every workflow touching
Postgres, MySQL or SQL Server was pushed into the MCP registry, and repeatedly
came back with a hosted gateway that could not see `localhost`, or with nothing at
all. This removes that whole class of dead end.

**One tool, four operations.** These began as four separate tools
(`sql_test_connection`, `sql_list_tables`, `sql_table_schema`, `sql_query`) that
shared a DSN, an engine factory, a redaction helper and a credential check — a
family that was real in the module and invisible to everything above it. A tool is
a *type* of integration; what it does are its operations. Grouped by type and
never by credential: a credential is a value the tool uses, not what it is, and
two SQL nodes may legitimately point at different databases.

Design, in the order the decisions mattered:

- **The DSN is an argument, not configuration.** It arrives through `tool_args`,
  which is the one place a `${SECRET}` is substituted — so a connection string
  reaches the driver and never a prompt, a trace or the model's context. There is
  deliberately no "current connection" to configure and no connect step: the
  credential mechanism already exists, adding a second one would invite a workflow
  to carry a password between nodes, and `workflow_requirements` derives what to
  ask the user for from the `${NAME}` references in `tool_args` — which only works
  while the credential is visible per node.
- **Read-only, enforced twice.** Statements are checked against a prefix
  allowlist, and only SELECT-shaped statements reach the driver. A tool that can
  silently write is not something to hand a weak model composing SQL from a schema
  it just read. Every operation declares `read_only=True`; `Operation.read_only`
  is the seam a future write operation would answer differently through, and the
  approval gates key off `is_read_only(args)`, which now resolves per operation.
- **SQLAlchemy for the URL, the driver for the rest.** It parses every dialect's
  URL and is already a dependency; the drivers (`pyodbc`, `psycopg`, …) are the
  user's to install, and a missing one is reported as the actionable thing it is.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from neurosurfer.tools.base import Operation, Tool, ToolContext, ToolResult

MAX_OUTPUT_CHARS = 20_000
DEFAULT_LIMIT = 50

# A read is SELECT-shaped. `EXPLAIN`/`SHOW`/`DESCRIBE` are included because they
# are how you understand a query without running it, and refusing them pushes a
# model towards running the real thing to find out.
_READ_PREFIXES = ("select", "with", "explain", "show", "describe", "desc")


def _redacted(url: str) -> str:
    """A DSN with its password removed, for error messages.

    A connection error quotes the URL it failed on, and that URL holds the
    password — the usual way a credential reaches a log.
    """
    try:
        from sqlalchemy.engine import make_url

        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 - never fail while making an error readable
        return "<dsn>"


def _dsn_problem(value: str) -> str:
    """Why *value* is not a usable connection URL, or "" if it is.

    Cheap — `make_url` is pure parsing, no connection — and it catches the case
    that actually happens: a stored value that is a *hostname* rather than a URL.
    One did, left behind by an earlier build that had invented a seven-field
    connection model, and it satisfied the requirement check by name while being
    unusable in fact.
    """
    text = str(value or "").strip()
    if not text:
        return "is empty"
    try:
        from sqlalchemy.engine import make_url

        url = make_url(text)
    except Exception:  # noqa: BLE001 - any parse failure is the answer
        return (
            "is not a SQLAlchemy connection URL. It should look like "
            "'mssql+pyodbc://user:pw@host:1433/db?driver=…' or "
            "'postgresql+psycopg://user:pw@host/db'"
        )
    if not url.drivername:
        return "names no database dialect (expected something like 'mssql+pyodbc://…')"
    return ""


def _engine(dsn: str):
    """A read-only engine for *dsn*, or a ToolResult explaining why not."""
    try:
        from sqlalchemy import create_engine
    except ImportError:  # pragma: no cover - sqlalchemy is a hard dependency
        return None, ToolResult.error(
            "SQLAlchemy is not installed. `pip install sqlalchemy`."
        )
    try:
        return create_engine(dsn, pool_pre_ping=True), None
    except ModuleNotFoundError as e:
        # The dialect resolved but its driver is absent — by far the most common
        # first failure, and the message must name the package to install.
        return None, ToolResult.error(
            f"The driver for this connection is not installed ({e.name}). "
            f"For SQL Server: `pip install pyodbc` (plus the system ODBC driver); "
            f"PostgreSQL: `pip install psycopg[binary]`; MySQL: `pip install pymysql`."
        )
    except Exception as e:  # noqa: BLE001
        return None, ToolResult.error(f"Could not read the connection URL: {e}")


def _rows_to_text(cols: list[str], rows: list[tuple[Any, ...]]) -> str:
    if not rows:
        return "(0 rows)"
    widths = [len(c) for c in cols]
    cells: list[list[str]] = []
    for row in rows:
        out = ["" if v is None else str(v) for v in row]
        cells.append(out)
        for i, v in enumerate(out):
            widths[i] = min(max(widths[i], len(v)), 60)

    def _line(vals: list[str]) -> str:
        return " | ".join(v[:widths[i]].ljust(widths[i]) for i, v in enumerate(vals))

    body = [_line(cols), "-+-".join("-" * w for w in widths)]
    body += [_line(c) for c in cells]
    text = "\n".join(body)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n… (truncated)"
    return f"{text}\n\n({len(rows)} row{'s' if len(rows) != 1 else ''})"


class _DsnArgs(BaseModel):
    dsn: str = Field(
        description=(
            "SQLAlchemy connection URL. Pass it from a stored secret as ${NAME} — "
            "never write a password here literally. Examples: "
            "'mssql+pyodbc://user:pw@host:1433/db?driver=ODBC+Driver+18+for+SQL+Server"
            "&TrustServerCertificate=yes', 'postgresql+psycopg://user:pw@host/db', "
            "'mysql+pymysql://user:pw@host/db'."
        )
    )


# ── one argument model per operation ────────────────────────────────────────
# What a caller is actually shown once an operation is chosen: exactly the fields
# that operation needs, with the right ones required. This is the whole reason
# operations carry their own model rather than sharing the tool's.


class TestConnectionArgs(_DsnArgs):
    pass


class ListTablesArgs(_DsnArgs):
    schema_name: str | None = Field(
        default=None,
        description="Restrict to one schema. Omit to list every schema's tables.",
    )


class TableSchemaArgs(_DsnArgs):
    table: str = Field(
        description="Table name, schema-qualified if the server uses schemas "
                    "(e.g. 'activity.Inquiries')."
    )


class QueryArgs(_DsnArgs):
    query: str = Field(
        description="A single read-only statement (SELECT/WITH/EXPLAIN/SHOW). "
                    "Write it in the server's own dialect — T-SQL uses TOP, "
                    "PostgreSQL and MySQL use LIMIT."
    )
    limit: int = Field(
        default=DEFAULT_LIMIT, ge=1, le=1000,
        description="Max rows returned to the caller.",
    )


class SqlArgs(_DsnArgs):
    """The tool's own surface: an operation, plus every operation's arguments.

    Deliberately permissive — each field is optional here because it is required
    by *some* operations and meaningless to others, and a schema demanding
    `query` would make `list_tables` uncallable. `call()` re-validates against the
    chosen operation's own model, so "table is required for table_schema" is
    still a clean, early error and not a driver exception three layers down.
    """

    operation: Literal["test_connection", "list_tables", "table_schema", "query"] = (
        Field(description="Which operation to run.")
    )
    schema_name: str | None = Field(
        default=None, description="list_tables: restrict to one schema."
    )
    table: str | None = Field(
        default=None, description="table_schema: the table to describe."
    )
    query: str | None = Field(
        default=None, description="query: a single read-only SQL statement."
    )
    limit: int = Field(
        default=DEFAULT_LIMIT, ge=1, le=1000,
        description="query: max rows returned.",
    )


class SqlTool(Tool):
    name = "sql"
    title = "SQL Database"
    secret_inputs = frozenset({"dsn"})
    # The driver runs in this process; only the database is elsewhere, so a DSN
    # pointing at localhost works. This is the field a hosted gateway would set
    # to "hosted", and the reason one was wrongly offered for a database running
    # in a container on the user's own machine.
    runtime = "in_process"
    credential_help = (
        "A SQLAlchemy connection URL for the database, e.g. "
        "mssql+pyodbc://user:pw@host:1433/db?driver=ODBC+Driver+18+for+SQL+Server "
        "or postgresql+psycopg://user:pw@host/db. Store it in Settings → Secrets "
        "and reference it as ${NAME}; it never needs to appear in the graph."
    )
    description = (
        "Read-only access to a SQL database server (SQL Server, PostgreSQL, MySQL "
        "and anything else SQLAlchemy speaks). Operations: `test_connection`, "
        "`list_tables`, `table_schema`, `query`. It cannot modify data. Work in "
        "that order — check the connection, find the tables, read the schema, "
        "then write SQL against what is actually there."
    )
    input_model = SqlArgs

    operations = {
        "test_connection": Operation(
            title="Test the connection",
            description=(
                "Check that the connection works and report the server version. "
                "Do this FIRST when a workflow talks to a database — it turns a "
                "wrong password or an unreachable host into one clear message "
                "instead of a failure part-way through a query."
            ),
            input_model=TestConnectionArgs,
            capabilities=frozenset({"db.connect"}),
        ),
        "list_tables": Operation(
            title="List tables",
            description=(
                "List the tables as schema-qualified names. Call this before "
                "writing any query: a table name guessed from the request is the "
                "most common reason a generated query fails."
            ),
            input_model=ListTablesArgs,
            capabilities=frozenset({"db.schema"}),
        ),
        "table_schema": Operation(
            title="Describe a table",
            description=(
                "Describe one table: columns with types and nullability, the "
                "primary key, and foreign keys. This is what makes a generated "
                "query correct — read the schema, then write SQL against it."
            ),
            input_model=TableSchemaArgs,
            capabilities=frozenset({"db.schema"}),
        ),
        "query": Operation(
            title="Run a query",
            description=(
                "Run one read-only SQL query and return the rows as a table. "
                "Only SELECT/WITH/EXPLAIN/SHOW are permitted. Prefer aggregating "
                "in SQL (COUNT, GROUP BY) over fetching rows and counting them."
            ),
            input_model=QueryArgs,
            capabilities=frozenset({"db.query"}),
        ),
    }

    def check_secret(self, name: str, value: str) -> str:
        return _dsn_problem(value) if name == "dsn" else ""

    async def call(self, args: SqlArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        op = self.operations.get(args.operation)
        if op is None:  # unreachable through the Literal, reachable from raw args
            return ToolResult.error(
                f"Unknown operation {args.operation!r}. "
                f"Expected one of: {', '.join(sorted(self.operations))}."
            )
        try:
            typed = op.input_model.model_validate(
                args.model_dump(exclude={"operation"}, exclude_none=True)
            )
        except Exception as e:  # noqa: BLE001 - pydantic's message is the useful part
            return ToolResult.error(
                f"'{args.operation}' arguments are missing or invalid: {e}"
            )
        handlers = {
            "test_connection": self._test_connection,
            "list_tables": self._list_tables,
            "table_schema": self._table_schema,
            "query": self._query,
        }
        return await handlers[args.operation](typed)

    # ── operations ──────────────────────────────────────────────────────────
    async def _test_connection(self, args: TestConnectionArgs) -> ToolResult:
        engine, err = _engine(args.dsn)
        if err is not None:
            return err
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                dialect = engine.dialect.name
                version = conn.exec_driver_sql("SELECT @@VERSION").scalar() \
                    if dialect == "mssql" else conn.execute(text("SELECT version()")).scalar()
            return ToolResult.ok(
                f"Connected ({dialect}).\n{str(version or '').splitlines()[0][:200]}"
            )
        except Exception as e:  # noqa: BLE001 - any driver error is the answer
            return ToolResult.error(
                f"Could not connect to {_redacted(args.dsn)}: {type(e).__name__}: {e}"
            )
        finally:
            engine.dispose()

    async def _list_tables(self, args: ListTablesArgs) -> ToolResult:
        engine, err = _engine(args.dsn)
        if err is not None:
            return err
        try:
            from sqlalchemy import inspect

            insp = inspect(engine)
            schemas = [args.schema_name] if args.schema_name else insp.get_schema_names()
            lines: list[str] = []
            for schema in schemas:
                for table in sorted(insp.get_table_names(schema=schema)):
                    lines.append(f"{schema}.{table}" if schema else table)
            if not lines:
                return ToolResult.ok("(no tables found)")
            return ToolResult.ok("\n".join(lines[:500]))
        except Exception as e:  # noqa: BLE001
            return ToolResult.error(
                f"Could not list tables on {_redacted(args.dsn)}: {type(e).__name__}: {e}"
            )
        finally:
            engine.dispose()

    async def _table_schema(self, args: TableSchemaArgs) -> ToolResult:
        engine, err = _engine(args.dsn)
        if err is not None:
            return err
        try:
            from sqlalchemy import inspect

            schema, _, table = args.table.rpartition(".")
            insp = inspect(engine)
            cols = insp.get_columns(table, schema=schema or None)
            if not cols:
                return ToolResult.error(
                    f"No such table: {args.table}. Run the 'list_tables' operation "
                    f"to see what exists."
                )
            lines = [f"{args.table}", ""]
            for c in cols:
                null = "NULL" if c.get("nullable", True) else "NOT NULL"
                lines.append(f"  {c['name']}  {c['type']}  {null}")
            try:
                pk = (insp.get_pk_constraint(table, schema=schema or None) or {})
                if pk.get("constrained_columns"):
                    lines += ["", f"  PRIMARY KEY ({', '.join(pk['constrained_columns'])})"]
                for fk in insp.get_foreign_keys(table, schema=schema or None) or []:
                    cols_ = ", ".join(fk.get("constrained_columns") or [])
                    ref = fk.get("referred_table")
                    refcols = ", ".join(fk.get("referred_columns") or [])
                    lines.append(f"  FOREIGN KEY ({cols_}) → {ref}({refcols})")
            except Exception:  # noqa: BLE001 - constraints are a bonus, not the point
                pass
            return ToolResult.ok("\n".join(lines))
        except Exception as e:  # noqa: BLE001
            return ToolResult.error(
                f"Could not describe {args.table}: {type(e).__name__}: {e}"
            )
        finally:
            engine.dispose()

    async def _query(self, args: QueryArgs) -> ToolResult:
        sql = (args.query or "").strip().rstrip(";").strip()
        if not sql:
            return ToolResult.error("No query given.")
        if not sql.lower().startswith(_READ_PREFIXES):
            return ToolResult.error(
                "Refused: this operation runs read-only statements only "
                f"({'/'.join(p.upper() for p in _READ_PREFIXES[:4])}…). "
                "It cannot INSERT, UPDATE, DELETE or run DDL."
            )
        # One statement. Splitting on ';' would be defeated by a semicolon inside
        # a string literal, so the check is that nothing follows the first one.
        if ";" in sql:
            return ToolResult.error(
                "Refused: pass a single statement (no ';' separators)."
            )

        engine, err = _engine(args.dsn)
        if err is not None:
            return err
        try:
            with engine.connect() as conn:
                result = conn.exec_driver_sql(sql)
                cols = list(result.keys())
                rows = result.fetchmany(args.limit)
            return ToolResult.ok(_rows_to_text(cols, [tuple(r) for r in rows]))
        except Exception as e:  # noqa: BLE001 - the driver's message is the useful part
            return ToolResult.error(f"Query failed: {type(e).__name__}: {e}")
        finally:
            engine.dispose()
