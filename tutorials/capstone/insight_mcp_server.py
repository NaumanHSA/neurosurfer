"""A tiny read-only SQLite MCP server for the Insight Engine capstone (stdio).

Exposes three read-only tools over the orders database so a graph `react` node
can answer ad-hoc questions in SQL. DB path comes from argv[1].
"""
import sqlite3
import sys

from mcp.server.fastmcp import FastMCP

server = FastMCP("sqlite-insight")
DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "insight.db"


@server.tool(annotations={"readOnlyHint": True})
def list_tables() -> str:
    """List the tables available in the database."""
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return ", ".join(r[0] for r in rows) or "(no tables)"
    finally:
        con.close()


@server.tool(annotations={"readOnlyHint": True})
def describe_table(table: str) -> str:
    """Return the column names and types of a table."""
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        if not rows:
            return f"No such table: {table}"
        return "\n".join(f"{r[1]} {r[2]}" for r in rows)
    finally:
        con.close()


@server.tool(annotations={"readOnlyHint": True})
def run_sql(query: str) -> str:
    """Run a read-only SELECT query and return up to 50 rows as text."""
    if not query.strip().lower().startswith("select"):
        return "Only SELECT queries are allowed."
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(50)
        header = " | ".join(cols)
        body = "\n".join(" | ".join(str(v) for v in r) for r in rows)
        return f"{header}\n{body}" if body else f"{header}\n(no rows)"
    except Exception as e:  # surface SQL errors so the agent can self-correct
        return f"SQL error: {e}"
    finally:
        con.close()


if __name__ == "__main__":
    server.run("stdio")
