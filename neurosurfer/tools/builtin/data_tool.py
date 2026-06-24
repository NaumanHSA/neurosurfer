"""Local structured-data tool — inspect/query CSV, JSON, JSONL, and SQLite files.

Read-only and dependency-free (stdlib ``csv``/``json``/``sqlite3``). Registered as a
read tool, so it honours the Task's ``path_deny`` guardrails. For SQLite it opens the
database **read-only** and refuses any statement that is not SELECT/WITH/PRAGMA/EXPLAIN,
so the tool can never mutate data.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult
from ..utils import resolve_path

MAX_OUTPUT_CHARS = 20_000
_READ_SQL_PREFIXES = ("select", "with", "pragma", "explain")


class DataArgs(BaseModel):
    path: str = Field(
        description="Path to a local .csv/.tsv, .json/.jsonl, or .sqlite/.db file."
    )
    query: str | None = Field(
        default=None,
        description=(
            "SQL SELECT for SQLite files, or a dotted key-path for JSON "
            "(e.g. 'results.0.name'). Ignored for CSV."
        ),
    )
    limit: int = Field(default=20, ge=1, le=1000, description="Max rows/items to return.")


class DataTool(Tool):
    name = "data"
    description = (
        "Inspect or query a local structured-data file. CSV/TSV → columns + preview rows; "
        "JSON → structure, or a value at a dotted key-path; JSONL → record preview; "
        "SQLite (.db/.sqlite) → run a read-only SQL SELECT. Read-only; cannot modify data."
    )
    input_model = DataArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: DataArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        path = resolve_path(ctx.cwd, args.path)
        if not path.exists():
            return ToolResult.error(f"File not found: {args.path}")
        if path.is_dir():
            return ToolResult.error(f"'{args.path}' is a directory, not a data file.")

        suffix = path.suffix.lower()
        try:
            if suffix in (".csv", ".tsv"):
                result = _read_csv(path, "\t" if suffix == ".tsv" else ",", args.limit)
            elif suffix == ".json":
                result = _read_json(path, args.query, args.limit)
            elif suffix in (".jsonl", ".ndjson"):
                result = _read_jsonl(path, args.limit)
            elif suffix in (".db", ".sqlite", ".sqlite3"):
                result = _query_sqlite(path, args.query, args.limit)
            else:
                return ToolResult.error(
                    f"Unsupported file type '{suffix}'. Supported: .csv, .tsv, .json, "
                    ".jsonl, .sqlite, .db"
                )
        except (OSError, UnicodeError) as e:
            return ToolResult.error(f"Could not read {args.path}: {e}")

        if not result.is_error and len(result.content) > MAX_OUTPUT_CHARS:
            return ToolResult.ok(result.content[:MAX_OUTPUT_CHARS] + "\n… [output truncated]")
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Format readers
# ──────────────────────────────────────────────────────────────────────────────
def _render_rows(columns: list[str], rows: list[list[Any]], note: str) -> str:
    head = " | ".join(columns)
    sep = "-" * min(max(len(head), 3), 120)
    body = "\n".join(
        " | ".join("" if c is None else str(c) for c in r) for r in rows
    )
    return f"{head}\n{sep}\n{body}\n\n({note})"


def _read_csv(path: Path, delimiter: str, limit: int) -> ToolResult:
    rows: list[list[str]] = []
    total = 0
    with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = next(reader, None)
        for row in reader:
            total += 1
            if len(rows) < limit:
                rows.append(row)
    if header is None:
        return ToolResult.ok(f"{path.name} is empty.")
    note = f"{len(rows)} of {total} data rows shown" if total > len(rows) else f"{total} data rows"
    table = _render_rows(header, rows, note)
    return ToolResult.ok(f"CSV {path.name} — {len(header)} columns\n\n{table}")


def _read_json(path: Path, query: str | None, limit: int) -> ToolResult:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if query:
        value, err = _descend(data, query)
        if err:
            return ToolResult.error(f"JSON path '{query}': {err}")
        data = value
    return ToolResult.ok(f"JSON {path.name}:\n\n{_render_json(data, limit)}")


def _read_jsonl(path: Path, limit: int) -> ToolResult:
    items: list[Any] = []
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            if len(items) < limit:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    items.append({"_unparsed": line[:200]})
    preview = json.dumps(items, indent=2, ensure_ascii=False)
    return ToolResult.ok(
        f"JSONL {path.name} — {total} records (showing {len(items)}):\n\n{preview}"
    )


def _query_sqlite(path: Path, query: str | None, limit: int) -> ToolResult:
    q = (query or "").strip().rstrip(";")
    if not q:
        return ToolResult.error("Provide a read-only SQL query (e.g. SELECT …) for SQLite files.")
    first = q.lower().split(None, 1)[0] if q else ""
    if first not in _READ_SQL_PREFIXES:
        return ToolResult.error(
            "Only read-only queries are allowed (SELECT / WITH / PRAGMA / EXPLAIN)."
        )
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return ToolResult.error(f"Cannot open SQLite database: {e}")
    try:
        cur = conn.execute(q)
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(limit + 1)
    except sqlite3.Error as e:
        return ToolResult.error(f"Query failed: {e}")
    finally:
        conn.close()

    more = len(fetched) > limit
    rows = [list(r) for r in fetched[:limit]]
    note = f"{len(rows)} rows shown" + (", more available" if more else "")
    if not columns:
        return ToolResult.ok(f"SQLite {path.name} — query OK ({note}).")
    return ToolResult.ok(f"SQLite {path.name}:\n\n{_render_rows(columns, rows, note)}")


# ──────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────────────────────
def _descend(data: Any, path_expr: str) -> tuple[Any, str | None]:
    """Walk ``data`` along a dotted path (list indices allowed). Returns (value, error)."""
    cur = data
    for part in path_expr.split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None, f"'{part}' is not a valid list index"
        elif isinstance(cur, dict):
            if part not in cur:
                return None, f"key '{part}' not found"
            cur = cur[part]
        else:
            return None, f"cannot descend into {type(cur).__name__} with '{part}'"
    return cur, None


def _render_json(value: Any, limit: int) -> str:
    if isinstance(value, list):
        head = value[:limit]
        suffix = f" (showing {len(head)})" if len(value) > len(head) else ""
        return f"list of {len(value)} items{suffix}:\n" + json.dumps(head, indent=2, ensure_ascii=False)
    if isinstance(value, dict):
        keys = ", ".join(str(k) for k in value)
        return f"object with {len(value)} keys ({keys}):\n" + json.dumps(value, indent=2, ensure_ascii=False)
    return json.dumps(value, indent=2, ensure_ascii=False)
