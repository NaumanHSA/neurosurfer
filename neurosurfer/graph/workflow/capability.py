"""Capability grounding: can a node actually do what it says it does?

:func:`validate_package` proved a graph was *well-formed* — the YAML parses, the DAG
is acyclic, every tool a node **names** exists. It never asked the question that
decides whether the workflow can run at all. A node whose goal is "read the file"
and whose tool list is empty is one LLM call being asked to produce the contents of
a file it cannot open. That graph validates clean and hallucinates at run time.

Two checks supply the missing question:

- :func:`toolless_react_node` — the structural half, and unambiguous. A ``react``
  node is *defined* as an LLM that calls tools in a loop; with no tools it is a
  ``base`` node in a costume. The executor used to hand it an empty pool, so it
  degraded into a model narrating actions it never took.
- :func:`suspected_capability` — the lexical half. Prompt text describing a reach
  outside the model (read a file, fetch a URL, check an inbox, send a message) on a
  node holding no tool.

Three rules keep the lexical half from becoming noise, each learned from a real
false positive on our own workflows:

1. **Verb→object scoping.** "read the file" is external, "read the summary" is not.
   Bare verbs match everything.
2. **The verb must open a clause.** *"Analyze the content read from the file"* is
   describing where its input came from, not asking to open anything. An action a
   node is being told to take starts a sentence or follows a connective; a verb
   sitting behind a noun is a reduced relative clause about provenance.
3. **Only fields that state an action are read.** ``expected_result`` describes the
   *output* — "List of important unread emails" is a noun phrase about shape, and
   scanning it flagged a node that filters a list it was handed.
4. **A wired-up node may already have been handed its data.** *"Analyze the content
   of the file"* on a node that ``depends_on`` the node which read that file is
   correct as written. This only forgives **source** capabilities (getting data in);
   no upstream node can send an SMS on your behalf, so sinks are always flagged.

Rule 4 is deliberately generous, and the trade is asymmetric: a missed warning
leaves us where we started, while a wrong one derails a build — on the run that
motivated it, gpt-4o-mini acknowledged the bogus warning correctly and then declared
the whole workflow blocked anyway.

The lexical half is reported as a *warning*, not an error, because the evidence is
still a string match: a package validated by a library caller should not be rejected
on a guess. The Architect's own registration gate escalates it to a refusal, where a
wrong call costs one extra tool round-trip instead of a rejected workflow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Capability",
    "suspected_capability",
    "node_action_text",
    "toolless_react_node",
    "upstream_may_satisfy",
    "GROUNDED_KINDS",
]

# Kinds that run a prompt and could hold tools. Everything else either takes no
# tools (`router`, `loop`, `map`, `subgraph`, `input`) or names one structurally
# (`tool`, `function`), where the engine already enforces its own requirement.
GROUNDED_KINDS = frozenset({"base", "react"})

# Short consonant-vowel-consonant stems that double before `-ing` (get → getting).
_DOUBLES = frozenset({"get", "run", "put", "hit", "scan", "log", "plan", "map"})


def _forms(verbs: str) -> str:
    """Expand ``read|fetch`` into every surface form a prompt actually uses.

    Node prompts say "Read the file", "Reads the file" and "Reading the file" in
    roughly equal measure; matching only the bare stem misses a third of them.
    Multi-word stems ("look up") are conjugated on their first word.
    """
    out: set[str] = set()
    for phrase in verbs.split("|"):
        phrase = phrase.strip()
        if not phrase:
            continue
        head, _, tail = phrase.partition(" ")
        tail = f" {tail}" if tail else ""
        if head.endswith("e"):
            variants = [head, head + "s", head[:-1] + "ing"]
        elif head.endswith("y") and len(head) > 2 and head[-2] not in "aeiou":
            variants = [head, head[:-1] + "ies", head + "ing"]
        elif head.endswith(("s", "x", "z", "ch", "sh")):
            variants = [head, head + "es", head + "ing"]
        elif head in _DOUBLES:
            variants = [head, head + "s", head + head[-1] + "ing"]
        else:
            variants = [head, head + "s", head + "ing"]
        out.update(v + tail for v in variants)
    # Longest first: the alternation must prefer `reading` over `read`.
    return "|".join(sorted(out, key=len, reverse=True))


# A verb only counts as an instruction when it opens a clause — sentence start, a
# bullet/number, or a connective. See rule 2 in the module docstring.
_CLAUSE_START = (
    r"(?:^|[.!?:;\n]\s*|[-*•]\s+|\d+[.)]\s*|"
    r"\b(?:to|and|then|or|but|must|should|shall|will|would|can|may|please|also|"
    r"first|next|finally|task is to|job is to|able to|needs? to|used to|in order to)\s+)"
)


@dataclass(frozen=True)
class Capability:
    """An external capability inferred from a node's prompt text."""

    label: str                  # human phrase, e.g. "read a file from disk"
    tools: tuple[str, ...]      # candidate registered tools that provide it
    #: Registry capability tags this need maps to. **Preferred over `tools`.**
    #:
    #: Naming tools here froze the answer to "what provides this" at the moment
    #: the taxonomy was written — so a capability with `tools=()` meant "nothing
    #: does this", even after somebody installed a server that did. A tag is
    #: resolved against the live registry instead, so an imported or authored
    #: tool answers the same need without this file changing.
    #:
    #: What is left here is the genuinely hard, genuinely curated part: mapping
    #: *plain English* to a tag. Mapping a tag to tools is the registry's job.
    tags: tuple[str, ...] = ()
    # Sources bring data IN (read a file, fetch a URL, check an inbox); sinks push
    # something OUT (send a message, write a file, run a command). The distinction
    # decides whether an upstream node could already have done the work — see
    # :func:`upstream_may_satisfy`.
    is_source: bool = True
    # Keywords to search the MCP registry with when no local tool provides this
    # (V3 Phase 2). The registry's search is effectively single-term — "read
    # gmail" returns nothing where "gmail" returns five servers — so these are
    # curated single words rather than anything derived from the phrase.
    registry_terms: tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.label


def _action(verbs: str, objects: str) -> re.Pattern[str]:
    """A clause-opening verb reaching an external object within the same sentence."""
    return re.compile(
        rf"{_CLAUSE_START}(?:{_forms(verbs)})\b[^.!?\n]{{0,40}}?\b(?:{objects})\b",
        re.IGNORECASE,
    )


# Ordered most-specific first; the first hit wins, so a node mentioning both an
# inbox and a file is reported against the capability its verb actually names.
_RULES: list[tuple[re.Pattern[str], Capability]] = [
    # ── email / inbox ──────────────────────────────────────────────────────────
    # "draft a reply to the email" is pure LLM work and must NOT match, so these
    # verbs are strictly about *reaching* a mailbox, never about composing.
    (
        _action(
            "read|fetch|check|monitor|retrieve|poll|scan|list|access|connect to"
            "|log into|watch|pull",
            r"e-?mails?|inbox|gmail|mailbox|imap|outlook|mail server",
        ),
        Capability("read an email inbox", (), tags=("inbox.read",),
                   registry_terms=("gmail", "email", "imap")),
    ),
    (
        _action(
            "send|deliver|dispatch|forward|email",
            r"e-?mails?|smtp|mail server|message to",
        ),
        Capability("send email", (), tags=("message.send",), is_source=False,
                   registry_terms=("email", "smtp", "sendgrid")),
    ),
    # ── messaging / notification ───────────────────────────────────────────────
    (
        _action(
            "send|deliver|dispatch|post|notify|alert|text|message|ping|push",
            r"sms|text ?messages?|slack|discord|telegram|whatsapp"
            r"|push notifications?|notifications?|webhooks?|pager",
        ),
        Capability("send a message or notification", (), tags=("message.send",),
                   is_source=False,
                   registry_terms=("sms", "twilio", "slack")),
    ),
    # ── rendering: a document or a picture, not prose ──────────────────────────
    # Both are capabilities this engine simply does not have, and saying so is the
    # point. Without them the fuzzy scorer answered "export data as a PDF file"
    # with `write_file` — which writes text, on a need whose whole distinguishing
    # word is PDF — and "generate charts" fell through to composition, producing a
    # `base` node that describes a chart instead of drawing one.
    (
        _action(
            "export|produce|generate|create|write|render|save|output|build",
            r"pdfs?|\.pdf\b|docx?|word documents?|reports? as|printable",
        ),
        Capability("produce a PDF document", (), tags=("pdf.render",),
                   registry_terms=("pdf", "document", "report")),
    ),
    (
        _action(
            "generate|create|produce|plot|draw|render|build|make",
            r"charts?|graphs?|plots?|visuali[sz]ations?|diagrams?|dashboards?"
            r"|bar charts?|line charts?|pie charts?",
        ),
        Capability("produce a chart or visualisation", (), tags=("chart.render",),
                   registry_terms=("chart", "plot", "visualization")),
    ),
    # ── filesystem ─────────────────────────────────────────────────────────────
    # Two noun-phrase rules first: "the contents of the file" states a reach into
    # the filesystem whatever verb introduces it.
    (
        re.compile(
            r"\bcontents?\s+of\s+(?:the\s+|a\s+|this\s+|that\s+)?"
            r"(?:file|document|pdf|csv|directory|folder)\b",
            re.IGNORECASE,
        ),
        Capability("read a file from disk", ("read_file",), tags=("file.read",)),
    ),
    (
        re.compile(
            r"\b(?:files?|documents?|pdfs?|csvs?)\s+"
            r"(?:located|stored|found|saved|sitting)\s+at\b",
            re.IGNORECASE,
        ),
        Capability("read a file from disk", ("read_file",), tags=("file.read",)),
    ),
    (
        _action(
            "read|open|load|parse|ingest|retrieve|access|get|extract",
            r"files?|documents?|pdfs?|csvs?|spreadsheets?|the file|source code",
        ),
        Capability("read a file from disk", ("read_file",), tags=("file.read",)),
    ),
    (
        _action(
            "list|walk|traverse|scan|enumerate|browse|index",
            r"director(?:y|ies)|folders?|file ?tree|files in",
        ),
        Capability("list a directory", ("list_dir",)),
    ),
    (
        _action(
            "write|save|store|persist|export|output|dump",
            r"to (?:a |the )?file|to disk|on disk|files?|\.txt|\.json|\.csv|\.md\b",
        ),
        Capability("write a file to disk", ("write_file",), is_source=False),
    ),
    # ── web / network ──────────────────────────────────────────────────────────
    (
        _action(
            "search|look up|google|research|browse|query|crawl",
            r"the web|the internet|online|google|search engines?",
        ),
        Capability("search the web", ("web_search",)),
    ),
    (
        _action(
            "fetch|download|scrape|crawl|request|call|hit|poll|retrieve|get|query",
            r"urls?|https?|apis?|endpoints?|web ?sites?|web ?pages?|rest\b|graphql",
        ),
        Capability("fetch a URL or call an API", ("http", "browse")),
    ),
    # ── shell / system ─────────────────────────────────────────────────────────
    (
        _action(
            "run|execute|invoke|launch|spawn",
            r"commands?|scripts?|shell|subprocess|binary|cli\b|terminal",
        ),
        Capability("run a shell command", ("run_command",), tags=("system.shell",),
                   is_source=False),
    ),
    # ── database ───────────────────────────────────────────────────────────────
    # Named engines come first, and each carries the terms an index actually
    # answers to. The generic rule below matched "query a SQL Server" and searched
    # postgres/database/sqlite — three engines the user had not asked for, and
    # never the one they had. A registry that holds `io.github.alyiox/mcp-mssql`
    # cannot return it to a search for "sqlite".
    (
        _action(
            "query|execute|run|insert into|read from|write to|connect to|select from",
            r"sql ?server|mssql|ms-sql|t-sql|transact-sql|azure sql",
        ),
        Capability("query a SQL Server database", ("sql",),
                   tags=("db.query", "db.schema", "db.connect"),
                   registry_terms=("mssql", "sqlserver", "sql-server")),
    ),
    (
        _action(
            "query|execute|run|insert into|read from|write to|connect to|select from",
            r"postgres(?:ql)?|\bpg\b|timescale|supabase",
        ),
        Capability("query a PostgreSQL database", ("sql",),
                   tags=("db.query", "db.schema", "db.connect"),
                   registry_terms=("postgres", "postgresql")),
    ),
    (
        _action(
            "query|execute|run|insert into|read from|write to|connect to|select from",
            r"mysql|mariadb",
        ),
        Capability("query a MySQL database", ("sql",),
                   tags=("db.query", "db.schema", "db.connect"),
                   registry_terms=("mysql", "mariadb")),
    ),
    (
        _action(
            "query|execute|run|insert into|read from|write to|connect to|select from",
            r"sqlite|\.db\b|\.sqlite3?\b",
        ),
        Capability("query a SQLite database", ("data", "sql"),
                   tags=("data.inspect", "db.query"),
                   registry_terms=("sqlite",)),
    ),
    (
        _action(
            "query|execute|run|insert into|read from|write to|connect to|select from",
            r"mongo(?:db)?|dynamo(?:db)?|cassandra|redis",
        ),
        Capability("query a NoSQL database", (),
                   registry_terms=("mongodb", "database")),
    ),
    (
        _action(
            "query|execute|run|insert into|read from|write to|connect to|select from",
            r"databases?|\bdb\b|sql|tables?",
        ),
        Capability("query a database", ("sql",),
                   tags=("db.query", "db.schema", "db.connect"),
                   registry_terms=("database", "sql")),
    ),
]


def suspected_capability(*texts: str | None) -> Capability | None:
    """The first external capability *texts* appear to describe, if any."""
    for text in texts:
        if not text:
            continue
        for pattern, capability in _RULES:
            if pattern.search(text):
                return capability
    return None


def node_action_text(node) -> tuple[str | None, ...]:
    """The strings on *node* that state what it is supposed to **do**.

    ``expected_result`` is deliberately excluded — it describes the output's shape,
    and "List of important unread emails" is a noun phrase, not an instruction to
    open a mailbox. The id is included with separators spaced out: a node literally
    named ``read_file`` states its job as plainly as its goal does, and in the
    failure that motivated this module it was the clearest signal on the node.
    """
    ident = re.sub(r"[_\-]+", " ", getattr(node, "id", "") or "")
    return (
        # `instructions` is the one field new nodes state their job in; the two
        # below are what nodes written before it used. All three are read,
        # because the graph on disk may be either shape.
        getattr(node, "instructions", None),
        getattr(node, "goal", None),
        getattr(node, "purpose", None),
        ident,
    )


def upstream_may_satisfy(node, capability: Capability) -> bool:
    """Could a node this one depends on already have done the fetching?

    A node wired to an upstream node is receiving its output, so "analyze the
    content of the file" reads as *the content I was handed*, not as an instruction
    to open anything. That only holds for **source** capabilities: no upstream node
    can send a message, write a file or run a command on this node's behalf, so
    sinks are flagged whatever the wiring looks like.
    """
    return capability.is_source and bool(getattr(node, "depends_on", None))


def toolless_react_node(node) -> bool:
    """A ``react`` node with an empty toolbelt — an agent that cannot act."""
    return getattr(node, "kind", None) == "react" and not (getattr(node, "tools", None) or [])
