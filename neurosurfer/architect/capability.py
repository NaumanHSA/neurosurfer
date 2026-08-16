"""The capability resolution ladder (Architect V3, Phase 2).

Phase 1 taught the validator to *notice* that a node needs a tool it doesn't have.
This is the other half: given a capability in plain English, find something that
actually provides it.

The ladder, in order, and it is walked **in code**:

1. **the live catalog** — built-in workflow tools, Architect-generated tools, and
   the tools of any MCP server currently connected. Anything here is usable now.
2. **the MCP registry** — thousands of servers we could install. A hit here is not
   a tool yet; it is a tool plus an install plus, usually, a credential.
3. **nothing** — and then the honest moves are `author_tool` or `declare_blocked`.

Doing the search deterministically is the point. Asking a small model "what tool
could read a file?" invites it to invent `file_reader`; there is a 60-entry alias
map in the tool registry that exists entirely because of that habit. Handing it five
real candidates ranked by a scorer asks a much easier question — *pick one* — and it
is the same question whether the model is strong or weak.

Nothing here installs, connects, or spends anything. It reads the catalog and
queries a public index; deciding is the agent's job and installing is gated on a
human.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from neurosurfer.mcp.registry import CredentialRequirement

__all__ = [
    "ToolCandidate",
    "ServerCandidate",
    "Resolution",
    "resolve_capability",
    "search_catalog",
]

# Words that carry no signal about *what a tool does*. Without this, "read a file"
# and "send a message" both match anything whose description contains "a".
_STOPWORDS = frozenset("""
a an the this that these those to of for from with without in on at by into over
and or but not no is are be been being it its my me i we you your our their his her
plus per via using use used uses suitable additional required
must should would could can may will shall need needs needed want wants please
any some all each every other another new old
""".split())

# Action verbs are the WEAK half of a capability phrase, and matching on them is
# how "read an email inbox" comes back with `read_file` — a suggestion that would
# be attached by a small model and produce fiction. What identifies a capability is
# its *object*: file, inbox, url, database. So verbs are stripped before a tool
# name is matched, and only object tokens decide whether a tool is a candidate.
_ACTION_VERBS = frozenset("""
read reads reading write writes writing send sends sending fetch fetches fetching
get gets getting put puts putting post posts posting list lists listing search
searches searching find finds finding open opens opening load loads loading save
saves saving store stores storing query queries querying call calls calling run
runs running execute executes executing check checks checking monitor monitors
monitoring poll polls polling scan scans scanning access accesses accessing
retrieve retrieves retrieving download downloads downloading upload uploads
uploading create creates creating delete deletes deleting update updates updating
notify notifies notifying alert alerts alerting connect connects connecting
""".split())

# Singular/plural collapse for the handful of shapes that actually collide here.
# A stemmer would be a dependency and a surprise; this is a dozen characters of
# behaviour we can read.
def _normalise(word: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(word) > 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _tokens(text: str) -> set[str]:
    words = re.split(r"[^a-z0-9]+", (text or "").lower())
    return {_normalise(w) for w in words if w and w not in _STOPWORDS and len(w) > 1}


def _objects(text: str) -> set[str]:
    """The tokens that identify *what* a capability acts on, verbs removed."""
    return {t for t in _tokens(text) if t not in _ACTION_VERBS}


# Obtaining a credential is not a capability — it is something the engine does.
# A node names `secrets: [NAME]` and the value is substituted into `tool_args` at
# call time; there is no step in between, and nothing to install for it.
#
# Without this the resolver treats "read environment variables at runtime" as a
# capability, finds nothing (correctly), and searches the registry — where the
# distinctive token is **environment**. A live build was offered the UK
# Environment Agency, a Swiss air-quality index, an Obsidian note searcher and
# IBM Quantum, and blocked on being unable to install any of them.
_CREDENTIAL_SOURCES = (
    r"env(?:ironment)?\s+var(?:iable)?s?|env\b|\.env\b"
    r"|credentials?|secrets?|api[\s_-]?keys?|tokens?|passwords?"
    r"|connection\s+(?:string|details|parameters|settings|config\w*)"
)
# …unless it is being read out of somewhere real. "Load the credentials from
# creds.json" IS a file read; the exclusion keeps this guard from swallowing one.
_CREDENTIAL_MEDIA = r"file|files|disk|path|json|yaml|yml|vault|s3|bucket|url"

_ENGINE_PROVIDED = re.compile(
    rf"\b(?:read|reads?|reading|load|loads?|loading|fetch|fetches|fetching"
    rf"|get|gets|getting|obtain|obtains?|resolve|resolves?|validate|validates?"
    rf"|access|accesses)\b[^.!?\n]{{0,30}}?\b(?:{_CREDENTIAL_SOURCES})\b",
    re.IGNORECASE,
)
_CREDENTIAL_MEDIUM_RE = re.compile(rf"\b(?:{_CREDENTIAL_MEDIA})\b", re.IGNORECASE)


def _is_engine_provided(need: str) -> bool:
    """Does *need* describe getting hold of a credential rather than using one?

    Answered before the taxonomy, because a phrase like "read env vars to fetch
    the SQL Server connection details" mentions a database and would otherwise
    resolve as one — when the truth is that it is not a step at all.
    """
    if not need or not _ENGINE_PROVIDED.search(need):
        return False
    return not _CREDENTIAL_MEDIUM_RE.search(need)


def _is_composition(need: str) -> bool:
    """Does *need* describe producing text rather than reaching for something?

    Only consulted after the curated taxonomy has declined to recognise the need,
    so "write a file to disk" is already settled as external before this is asked
    — the check is what remains once no external object was found.

    **The verb is not enough.** It used to be, and "generate visual charts" and
    "analyze product sales data" were both declared internal on the strength of
    `generate` and `analyze` — one is matplotlib and the other is a database. A
    composition verb only means composition when what it acts on is text, so an
    object naming a medium or a data source disqualifies it.
    """
    words = [w for w in re.split(r"[^a-z]+", (need or "").lower()) if w]
    if not any(w in _COMPOSITION_VERBS for w in words):
        return False
    return not (_objects(need) & _NON_TEXT_OBJECTS)


def _raw_objects(text: str) -> list[str]:
    """Object words as written, for use as *search queries* rather than matches.

    ``_normalise`` exists to make "file" and "files" compare equal; it is a crude
    suffix strip and it mangles words that merely end in a plural-looking suffix —
    "kubernetes" becomes "kubernet", which matches nothing in any index. Fine for
    comparing, useless for querying.
    """
    words = re.split(r"[^A-Za-z0-9]+", (text or "").lower())
    seen: list[str] = []
    for word in words:
        if (word and len(word) > 1 and word not in _STOPWORDS
                and word not in _ACTION_VERBS and word not in seen):
            seen.append(word)
    return seen


@dataclass
class ToolCandidate:
    """A tool that exists right now and can be assigned to a node today."""

    name: str
    description: str
    origin: str                 # builtin | generated | mcp
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description[:200],
                "origin": self.origin}


@dataclass
class ServerCandidate:
    """An MCP server that would provide the capability once installed."""

    name: str
    description: str
    credentials: list[CredentialRequirement] = field(default_factory=list)
    runs_locally: bool = False
    runs_remotely: bool = False
    installed: bool = False
    score: float = 0.0

    @property
    def required_credentials(self) -> list[CredentialRequirement]:
        return [c for c in self.credentials if c.required]

    @property
    def blocking_credentials(self) -> list[CredentialRequirement]:
        """Required credentials this machine does not already have.

        The distinction `required_credentials` cannot make: an entry declaring
        ``Authorization: "Bearer {smithery_api_key}"`` is *required* and is also
        already satisfied whenever that key is on file. Blocking a build on the
        first is how it came to ask for a value it was holding.
        """
        from neurosurfer.mcp.credentials import blocking

        return blocking(self.credentials)

    def to_dict(self) -> dict[str, Any]:
        from neurosurfer.mcp.credentials import asked_name, satisfied_from

        return {
            "name": self.name,
            "description": self.description[:200],
            "installed": self.installed,
            "runs_locally": self.runs_locally,
            "runs_remotely": self.runs_remotely,
            "credentials": [
                # `ask_for` is what a person supplies; `satisfied_from` is why a
                # required credential may still not be in the way.
                {**c.to_dict(), "ask_for": asked_name(c),
                 "satisfied_from": satisfied_from(c)}
                for c in self.credentials
            ],
        }


# Verbs describing work a language model does by thinking, not by reaching. Asked
# to resolve "draft an email", the ladder used to search the registry and cheerfully
# return five email servers — and gpt-4o-mini then reported "drafting emails" as a
# credentialled blocker. Composing text is the one thing a `base` node is *for*.
_COMPOSITION_VERBS = frozenset("""
draft drafts drafting write writes writing compose composes composing summarise
summarises summarising summarize summarizes summarizing classify classifies
classifying analyse analyses analysing analyze analyzes analyzing rewrite rewrites
rewriting translate translates translating judge judges judging review reviews
reviewing evaluate evaluates evaluating score scores scoring rank ranks ranking
generate generates generating explain explains explaining answer answers answering
""".split())

# Objects a composition verb cannot be composing: a medium it would have to render
# into, or a source it would have to reach for. Hand-checked and small, like the
# curated taxonomy it sits beside — the alternative is a scorer guessing at the
# difference between "write a summary" and "write to the database", and a wrong
# guess here is a `tool` node with no tool.
#
# Stemmed to match `_normalise`, which is why it is `sal` and not `sales`.
_NON_TEXT_OBJECTS = frozenset("""
chart charts graph graphs plot plots figure figures diagram diagrams
image images picture pictures photo photos visual visuals dashboard
pdf csv xlsx spreadsheet spreadsheets workbook
database databases db sql table tables query queries schema
api endpoint endpoints url inbox mailbox repository repositories
file files disk directory folder bucket warehouse
sal revenue transaction transactions record records row rows dataset
""".split())


@dataclass
class Resolution:
    """What the ladder found for one capability."""

    need: str
    tools: list[ToolCandidate] = field(default_factory=list)
    servers: list[ServerCandidate] = field(default_factory=list)
    registry_searched: bool = False
    registry_error: str = ""
    # The curated taxonomy recognised this capability and states that nothing
    # built in provides it — a stronger claim than "the search found nothing".
    curated_gap: bool = False
    # This isn't an external capability at all; it is what a `base` node does.
    no_tool_needed: bool = False

    @property
    def status(self) -> str:
        """``not_external`` — a base node does it · ``have`` — usable now ·
        ``installable`` — one install away · ``none``."""
        if self.no_tool_needed:
            return "not_external"
        if self.tools:
            return "have"
        if self.servers:
            return "installable"
        return "none"

    def to_dict(self) -> dict[str, Any]:
        """The serialized form carried on a plan step.

        One definition rather than one per caller: the planner built this inline
        and the post-install re-check needed the same shape, which is exactly how
        two subtly different resolutions end up on the same field.
        """
        return {
            "need": self.need,
            "status": self.status,
            "tools": [t.to_dict() for t in self.tools],
            "servers": [s.to_dict() for s in self.servers],
            "curated_gap": self.curated_gap,
            # What was *searched*, not only what was chosen (V3 Phase 6). When the
            # Architect lands on the wrong tool — or on nothing — this is the
            # difference between seeing the outcome and seeing the reasoning.
            "registry_searched": self.registry_searched,
            "registry_error": self.registry_error,
        }

    def render(self) -> str:
        """The text the agent reads. Ends by naming the move it should make."""
        lines = [f"Capability: {self.need}", f"Status: {self.status.upper()}", ""]

        if self.no_tool_needed:
            return "\n".join(lines + [
                "This is not an external capability — it is writing, and writing is "
                "what an LLM step does. No tool is needed and nothing is blocking.",
                "",
                "NEXT:",
                "  Make it a `base` node with a clear purpose/goal. Do NOT install a "
                "server and do NOT declare_blocked for this step.",
            ])

        if self.tools:
            lines.append("Tools you can assign RIGHT NOW:")
            for t in self.tools:
                lines.append(f"  • `{t.name}` ({t.origin}) — {t.description[:150]}")
        elif self.curated_gap:
            # This used to end "…it needs an external integration, NOT a catalog
            # tool", which reads as "do not write one" — and was wrong about
            # databases for the entire life of the project, because the catalog
            # simply had no database tool until one was added.
            lines.append(
                "No built-in tool provides this. It may still be authorable — "
                "see NEXT."
            )
        else:
            lines.append("No tool in the catalog provides this.")

        if self.servers:
            lines += ["", "MCP servers that would provide it (each needs installing):"]
            for s in self.servers:
                mark = " [already installed]" if s.installed else ""
                lines.append(f"  • `{s.name}`{mark} — {s.description[:150]}")
                creds = s.blocking_credentials
                if creds:
                    # Capped: one real entry declares eleven env vars, and a wall
                    # of them buries the two that are actually secrets.
                    shown = "; ".join(c.render() for c in creds[:5])
                    extra = f" (+{len(creds) - 5} more)" if len(creds) > 5 else ""
                    lines.append(f"      needs: {shown}{extra}")
                elif s.required_credentials:
                    # Declared, and already held. Saying so stops the agent
                    # treating a satisfied requirement as a reason to give up.
                    lines.append(
                        "      needs: nothing further — its credentials are "
                        "already configured here"
                    )
                else:
                    lines.append("      needs: no credentials declared")
        elif self.registry_searched and not self.tools:
            lines += ["", "Nothing in the MCP registry matched either."]
        if self.registry_error:
            lines += ["", f"(Registry search failed: {self.registry_error})"]

        lines += ["", "NEXT:"]
        if self.status == "have":
            lines.append(
                "  Assign one of the tools above — `update_node` with "
                "`kind: 'tool'` for a single call, or `kind: 'react'` with `tools`."
            )
        elif self.status == "installable":
            # Both routes, always. Listing only the install is what made
            # `author_tool` unreachable in practice: `installable` means the
            # registry returned *something*, however irrelevant, so a single junk
            # match hid authoring completely. One build was offered the UK
            # Environment Agency for "read environment variables", declined to
            # install it — correctly — and blocked, never having been told that
            # writing the tool was an option.
            lines += [
                "  Two routes. Choose by what the capability actually needs:",
                "",
                "  • `author_tool` — when it is plain Python plus a value the user "
                "can paste (a connection string, an API key): a database query, a "
                "file conversion, a REST call, a calculation. This is also the "
                "safer route: an authored tool is generated here, sandbox-tested "
                "and approved, where installing runs someone else's code.",
                "  • `install_mcp_server` — when it needs a vendor account, a "
                "browser sign-in, an OAuth grant or a proprietary SDK. Reproducing "
                "an auth flow in an authored tool does not work.",
                "",
                "  If neither fits, `declare_blocked` and list exactly what the "
                "user must provide.",
            ]
        else:
            lines.append(
                "  Nothing in the catalog or the registry provides this. "
                "`author_tool` if it can be built from Python (it usually can — a "
                "few dozen lines against a documented API or library), otherwise "
                "`declare_blocked` saying precisely what is missing."
            )
        return "\n".join(lines)


def _origins() -> tuple[set[str], set[str]]:
    """(generated tool names, live MCP tool names) — for labelling only."""
    try:
        from neurosurfer.tools.generated import load_generated_tools
        from neurosurfer.tools.registry import live_tools

        return {t.name for t in load_generated_tools()}, {t.name for t in live_tools()}
    except Exception:  # noqa: BLE001 - origin is a label, never a hard failure
        return set(), set()


def _candidate(tool, generated: set[str], live: set[str], score: float) -> ToolCandidate:
    description = (getattr(tool, "description", "") or "").strip()
    origin = ("mcp" if tool.name in live
              else "generated" if tool.name in generated else "builtin")
    return ToolCandidate(
        name=tool.name, description=description.split("\n")[0],
        origin=origin, score=score,
    )


def search_catalog(need: str, *, limit: int = 5) -> list[ToolCandidate]:
    """Rank the live workflow-usable catalog against *need*.

    A tool is a candidate only when its **name** shares an object token with the
    need — not a verb, and not merely a word from its description. The looser
    version of this returned `read_file` for "read an email inbox" and `http` for
    "send a text message" (its description mentions "text"), which is worse than
    returning nothing: a small model attaches the suggestion and ships fiction.

    Built-ins, generated tools and connected MCP tools compete on the same footing,
    because from a node's point of view they are the same thing.
    """
    from neurosurfer.tools.registry import workflow_node_tools

    want = _objects(need)
    if not want:
        return []

    try:
        tools = workflow_node_tools()
    except Exception:  # noqa: BLE001 - a broken registry must not break resolution
        return []
    generated, live = _origins()

    scored: list[ToolCandidate] = []
    for tool in tools:
        name_hits = want & _objects(tool.name)
        if not name_hits:
            continue
        description = (getattr(tool, "description", "") or "").strip()
        if not _name_is_specific_enough(tool.name, want, name_hits, description):
            continue
        # The object gates; the verb ranks. `read_file` and `write_file` both hold
        # "file", so without the verb they tie and the shortlist for "read a file"
        # opens with a coin toss. Description overlap only breaks remaining ties.
        verb_hits = _tokens(need) & _tokens(tool.name) & _ACTION_VERBS
        score = len(name_hits) * 3.0 + len(verb_hits) * 2.0 + len(want & _tokens(description))
        scored.append(_candidate(tool, generated, live, score))
    scored.sort(key=lambda c: (-c.score, c.name))
    return scored[:limit]


def _name_is_specific_enough(
    name: str, want: set[str], name_hits: set[str], description: str
) -> bool:
    """Is a name match strong enough to stand on its own?

    `read_file` is two tokens and one of them is a verb: matching "file" there is
    a claim about *reading files*. `data` is a single generic noun, so it matches
    at full strength any need that happens to contain the word — which is how
    "export data as PDF", "query customer revenue data" and "query sales data for
    comparison" all resolved to a local-CSV inspector, reported `have`, and
    therefore blocked nothing.

    So a single-token name has to be corroborated: if the need names other objects,
    at least one of them must appear in the tool's own description. `data` keeps
    "query a CSV data file" (its description lists CSV) and loses the PDF export.
    """
    if len(_tokens(name)) > 1:
        return True
    others = want - name_hits
    if not others:
        # The need is *only* the generic word. Nothing better will match either.
        return True
    return bool(others & _tokens(description))


def _live_hits(need: str, limit: int) -> list[ToolCandidate]:
    """Connected MCP tools that could serve *need*, matched on description too.

    Looser than `search_catalog` in *what* it reads — the publisher's description
    counts, not only the name, because "Execute a SELECT query against PostgreSQL"
    is far better evidence than a name like `query`.

    It is **not** looser about what counts as a match, and that distinction was
    learned the hard way. The original version scored any shared token, on the
    theory that a live tool is here because somebody installed a server *for this
    build*. That prior dies the moment a server outlives the build that installed
    it: a Postgres server left connected from an earlier run answered "generate
    chart images using a charting library" with `generate_migration` and
    `generate_seed_data`, on the strength of the word **generate** — and
    `saved_queries`, because a *query* library matched a *charting* library. Those
    resolved to `have`, so nothing blocked, nothing offered an install, and five
    nodes were built to draw charts with database-migration tools.

    So a match must be about the *thing*, never the action:

    - **Verbs are not evidence.** Action verbs are already stripped by `_objects`;
      composition verbs (`generate`, `analyze`, `write`) are stripped here too.
      They describe what is being done, and almost every tool does something.
    - **One loose token is not enough.** Either the tool's *name* names one of the
      need's objects, or its *description* names at least two. A single incidental
      word shared with a description is the failure mode above.
    """
    from neurosurfer.tools.registry import live_tools

    def _significant(text: str) -> set[str]:
        return _objects(text) - _COMPOSITION_VERBS

    want = _significant(need)
    if not want:
        return []
    generated, live_names = _origins()
    scored: list[ToolCandidate] = []
    for tool in live_tools():
        description = (getattr(tool, "description", "") or "").strip()
        name_hits = want & _significant(tool.name)
        desc_hits = want & (_tokens(description) - _COMPOSITION_VERBS)
        # The corroboration gate. A name match is direct evidence; a description
        # match is circumstantial, so it takes two.
        if not name_hits and len(desc_hits) < 2:
            continue
        scored.append(
            _candidate(tool, generated, live_names,
                       len(name_hits) * 3.0 + len(desc_hits))
        )
    scored.sort(key=lambda c: (-c.score, c.name))
    return scored[:limit]


def _curated_tools(need: str, limit: int) -> tuple[list[ToolCandidate], bool] | None:
    """What the Phase 1 taxonomy already knows about this capability.

    ``capability.py`` in the graph layer classifies prompt text into a curated set
    of external capabilities, each with the tools that provide it. That mapping is
    hand-checked and tested; re-deriving it with a fuzzy scorer would be both
    redundant and worse. Returns ``(candidates, curated_empty)`` — where
    ``curated_empty`` means *we know nothing built in does this*, which is a fact
    worth acting on rather than a search that came up short.
    """
    from neurosurfer.graph.workflow.capability import suspected_capability
    from neurosurfer.tools.registry import workflow_node_tools

    known = suspected_capability(need)
    if known is None:
        return None

    # Tags first. The taxonomy's job is plain English → capability; deciding
    # *which tool* has that capability is the registry's, and asking it live is
    # what lets a server imported this morning answer a need written months ago.
    # A hardcoded `tools=()` used to mean "nothing does this" forever.
    if known.tags:
        from neurosurfer.registry import providers_of

        names: list[str] = []
        for tag in known.tags:
            names += [m.name for m in providers_of(tag) if m.name not in names]
        if names:
            pool = {t.name: t for t in workflow_node_tools()}
            generated, live = _origins()
            found = [
                _candidate(pool[n], generated, live, 100.0) for n in names if n in pool
            ]
            if found:
                return found[:limit], False
        # Nothing declares the tag. That is a *fact* now, not a search that came
        # up short — and the honest next move is to author or import, which is
        # what `curated_gap` tells the model.
        live_hits = _live_hits(need, limit)
        return (live_hits, False) if live_hits else ([], True)

    if not known.tools:
        # The taxonomy names *built-ins*. It cannot know what a server installed
        # thirty seconds ago exposes — and filling this exact gap is what an
        # install is for. Returning "nothing does this" without looking is why a
        # build installed a Postgres server, re-resolved, was told `installable`
        # again, installed the same server a second time, and then gave up.
        live = _live_hits(need, limit)
        return (live, False) if live else ([], True)
    try:
        tools = {t.name: t for t in workflow_node_tools()}
    except Exception:  # noqa: BLE001
        return None
    generated, live = _origins()
    found = [
        _candidate(tools[name], generated, live, 100.0)
        for name in known.tools if name in tools
    ]
    return (found[:limit], False) if found else ([], True)


def registry_terms(need: str, *, semantic: bool = False) -> list[str]:
    """Queries to search the registry with, most specific first.

    A keyword index behaves as a *single-term* index: ``"read gmail"`` returns
    nothing while ``"gmail"`` returns five servers, and ``"query postgres
    database"`` returns nothing while ``"postgres"`` returns five. So against one
    of those the query is never the phrase.

    Against a **semantic** engine it is exactly the phrase, and reducing it to
    keywords throws away the meaning that engine exists to read. `semantic` is set
    from the source's declared `Capability.SEMANTIC_SEARCH`, never from its name.

    When the taxonomy recognises the capability its curated terms lead — they
    encode that an inbox is reached through *gmail* or *imap*, which no amount of
    tokenising "read an email inbox" would tell you. They no longer *replace* the
    need's own words: curated terms for "query a database" are postgres/database/
    sqlite, so "query a **SQL Server**" used to search for three things the user
    had not asked for and never for the one they had.
    """
    if semantic:
        return [need.strip()] if need.strip() else []

    from neurosurfer.graph.workflow.capability import suspected_capability

    known = suspected_capability(need)
    curated = list(known.registry_terms) if known is not None else []
    # Longest first as a rough proxy for specificity: `postgres` before `db`.
    own = sorted(_raw_objects(need), key=lambda t: (-len(t), t))
    terms: list[str] = []
    for term in [*curated, *own]:
        if term and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
    # Capped: each term is a separate round trip, and past the fourth they are
    # generic enough ("data", "results") to add noise rather than candidates.
    return terms[:4]


def search_servers(need: str, *, limit: int = 5) -> tuple[list[ServerCandidate], str]:
    """Rank MCP servers against *need*, on the engine chosen for this context.

    Returns (candidates, error). The engine comes from `active_source()` — this
    used to import the official client directly, so choosing Smithery changed what
    the catalog browser showed and nothing about what a build searched.
    """
    from neurosurfer.mcp.registry import McpRegistryError
    from neurosurfer.mcp.sources import Capability as SourceCapability
    from neurosurfer.mcp.sources import SourceUnavailable, active_source

    source = active_source()
    semantic = SourceCapability.SEMANTIC_SEARCH in getattr(source, "capabilities", frozenset())
    terms = registry_terms(need, semantic=semantic)
    if not terms:
        return [], ""

    # Concurrently: each term is a separate several-second round trip to the public
    # registry, and three of them in sequence made resolving one capability an
    # ~18-second pause in the middle of a build. They are independent queries.
    def _one(term: str) -> tuple[list[Any], str]:
        try:
            return source.search(term, limit=10), ""
        except SourceUnavailable as e:
            # The engine is selected but not configured. Say so — degrading to the
            # default silently is how a user concludes their choice did nothing.
            return [], str(e)
        except McpRegistryError as e:
            return [], str(e)
        except Exception as e:  # noqa: BLE001 - discovery is best-effort, never fatal
            return [], f"{type(e).__name__}: {e}"

    hits: list[Any] = []
    seen: set[str] = set()
    error = ""
    with ThreadPoolExecutor(max_workers=len(terms)) as pool:
        # Term order is preserved — it encodes specificity (`postgres` before `db`),
        # and the ranking downstream leans on it.
        for found, err in pool.map(_one, terms):
            error = error or err
            for hit in found:
                if hit.name not in seen:
                    seen.add(hit.name)
                    hits.append(hit)
    if not hits:
        return [], error

    installed: set[str] = set()
    try:
        from neurosurfer.config.mcp import McpStore

        installed = {c.name for c in McpStore.default().list()}
    except Exception:  # noqa: BLE001
        pass

    # Rank by how much of the need — and of the curated terms that found it — the
    # entry's own words cover. Keyword search over a registry with no quality
    # signal cannot be precise, so this orders a shortlist rather than picking a
    # winner: the agent reads the descriptions and chooses, which is the one part
    # of this a language model is reliably better at than a scorer.
    #
    # A semantic engine has already done the ranking, and by meaning rather than by
    # shared words. Re-scoring lexically there is not a refinement, it is a second
    # opinion from the weaker judge — and the `score <= 0` cut would *discard* an
    # exactly-right server that happens to share no tokens with the phrase. So on a
    # semantic source the engine's own order stands and nothing is dropped.
    want = _tokens(need) | ({t.lower() for t in terms} if not semantic else set())
    scored: list[ServerCandidate] = []
    for rank, hit in enumerate(hits):
        if semantic:
            # Descending with position, so the engine's first hit stays first.
            score = float(len(hits) - rank)
        else:
            score = len(want & _tokens(hit.name)) * 2.0 + len(want & _tokens(hit.description))
            if score <= 0:
                continue
        local_name = hit.name.split("/")[-1]
        scored.append(ServerCandidate(
            name=hit.name,
            description=hit.description,
            credentials=list(hit.credentials),
            runs_locally=hit.runs_locally,
            runs_remotely=hit.runs_remotely,
            installed=local_name in installed,
            score=score,
        ))
    # An installed server first: it is already a decision this deployment made.
    scored.sort(key=lambda c: (not c.installed, -c.score, c.name))
    return scored[:limit], ""


def resolve_capability(
    need: str, *, limit: int = 5, include_registry: bool = True
) -> Resolution:
    """Walk the ladder for one plain-English capability.

    The registry is only consulted when the catalog comes up empty — if a tool
    already does the job, offering an install alongside it invites the agent to
    take the expensive path for no reason.
    """
    need = (need or "").strip()
    resolution = Resolution(need=need)
    if not need:
        return resolution

    # 0. Is this a credential being obtained rather than used? Answered before
    #    the taxonomy, because such a phrase usually names the thing it is a
    #    credential *for* ("read env vars to get the SQL Server connection") and
    #    would otherwise resolve as a database capability — or, when it names
    #    nothing, be searched for in a public registry on the word "environment".
    if _is_engine_provided(need):
        resolution.no_tool_needed = True
        return resolution

    # 1. The curated taxonomy first. When it recognises the capability its answer
    #    is authoritative in both directions — including "nothing built in does
    #    this", which sends us straight to the registry instead of offering the
    #    nearest-looking file tool.
    curated = _curated_tools(need, limit)
    if curated is not None:
        resolution.tools, curated_empty = curated
    else:
        curated_empty = False
        # 1b. The taxonomy didn't recognise an external capability. If the need
        #     is phrased as composition, say so plainly rather than searching —
        #     "draft an email" otherwise returns email servers, and a small model
        #     reads that as a credentialled blocker on a step that needs nothing.
        if _is_composition(need):
            resolution.no_tool_needed = True
            return resolution
        # 2. Otherwise fall back to a strict catalog search.
        resolution.tools = search_catalog(need, limit=limit)

    if resolution.tools or not include_registry:
        return resolution

    # 3. Nothing local. Ask the registry what could provide it.
    servers, error = search_servers(need, limit=limit)
    resolution.servers = servers
    resolution.registry_searched = True
    resolution.registry_error = error
    resolution.curated_gap = curated_empty
    return resolution
