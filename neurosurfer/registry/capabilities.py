"""The controlled vocabulary a tool declares itself against.

Deliberately small, hand-written, and closed. The alternative — letting each tool
invent its own tags — re-admits the matching problem one level up: two tools
tagged `db.query` and `sql.run` would need a scorer to reconcile, and a scorer
guessing at tags is what this whole layer exists to replace.

A tag is a *capability*, not a tool and not an implementation. `sql.query` is
"run a read-only query against a database server" whoever provides it — a core
tool, an imported MCP server, or something authored last Tuesday. That is the
property that lets resolution stop caring where a tool came from.

Growth is expected and cheap: add the constant, tag the tools, map the phrases.
What is not cheap is a tag that means two things, so each one carries a sentence
saying exactly what it covers.
"""

from __future__ import annotations

__all__ = ["CAPABILITIES", "describe", "is_known", "unknown"]


#: tag → what a tool claiming it must actually be able to do.
CAPABILITIES: dict[str, str] = {
    # ── files on disk ──────────────────────────────────────────────────────
    "file.read": "Read the contents of a file at a given path.",
    "file.write": "Create or overwrite a file at a given path.",
    "file.edit": "Change part of an existing file, leaving the rest alone.",
    "file.list": "List directory entries, or glob for paths.",
    "file.search": "Search file *contents* by pattern and return matches.",
    # ── structured data ────────────────────────────────────────────────────
    "data.inspect": "Inspect or query a structured data FILE (CSV, JSON, SQLite).",
    # ── a database server, which is not a file ─────────────────────────────
    "db.connect": "Prove a database connection works and report what answered.",
    "db.schema": "Discover a database's tables, columns, types and keys.",
    "db.query": "Run a read-only SQL query against a database server.",
    # ── the network ────────────────────────────────────────────────────────
    "web.request": "Make an HTTP request to a URL and return the response.",
    "web.browse": "Render a page in a browser and return its readable text.",
    "web.search": "Search the web and return results with URLs.",
    # ── this machine ───────────────────────────────────────────────────────
    "system.shell": "Run a shell command and return its output and exit code.",
    # No `system.python`: a workflow runs Python through a `function` node, not a
    # tool, so listing it as an unprovided capability told the model it could not
    # do something it can. The vocabulary covers what a *node* resolves against.
    # ── producing artifacts, which nothing here does yet ───────────────────
    # Declared before anything claims them, because the resolver's honest answer
    # to "generate a chart" should be "no tool has this capability" rather than
    # whichever tool shares the most words with the request. Naming the gap is
    # what turns it into an authorable or importable job.
    "chart.render": "Render data as a chart image.",
    "pdf.render": "Produce a PDF document from text, HTML or images.",
    # ── talking to people ──────────────────────────────────────────────────
    "message.send": "Send a message to a person or channel (email, chat, SMS).",
    "inbox.read": "Read messages from a mailbox or channel.",
}


def is_known(tag: str) -> bool:
    return tag in CAPABILITIES


def describe(tag: str) -> str:
    return CAPABILITIES.get(tag, "")


def unknown(tags) -> list[str]:
    """Tags outside the vocabulary — a freshness test turns these into failures."""
    return sorted({t for t in (tags or ()) if t not in CAPABILITIES})
