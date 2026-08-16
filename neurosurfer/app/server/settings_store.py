"""SQLite-backed per-account settings: provider profiles and UI preferences.

Two tables in one file beside ``auth.db``, for the same reasons: a single-process
self-hosted gateway, where an external database would be infrastructure without a
benefit and the file is trivially backed up.

**Why per account rather than host-level.** MCP servers are host-level on purpose —
they are processes the gateway spawns as its own OS user, so a per-account copy
would look like isolation while providing none. Provider profiles are the opposite:
an API key is a personal credential billed to a person, and two users of one gateway
must not spend each other's quota. They follow the account, as workspaces do.

**Keys are write-only.** :meth:`ProviderRow.public` never emits ``api_key``, only a
mask. A studio that can show you your own key is a studio that hands it to any script
running on the origin; the key is needed by the *server* to make calls, not by the
browser to render a form.

The library keeps its own JSON :class:`~neurosurfer.config.profiles.ProviderStore`
for CLI and embedded use — the engine must not require the server's database. This
store is the gateway's, and it imports the JSON one once, on first run.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurosurfer.config.profiles import ProviderProfile, mask_secret

# The identity-free caller — library, CLI, an unauthenticated gateway — lands here,
# exactly as it lands in the `default` workspace.
DEFAULT_OWNER = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_profiles (
    owner       TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    kind        TEXT    NOT NULL DEFAULT 'openai',
    base_url    TEXT,
    model       TEXT    NOT NULL DEFAULT '',
    api_key     TEXT,
    context_window   INTEGER NOT NULL DEFAULT 200000,
    max_output_tokens INTEGER NOT NULL DEFAULT 8192,
    supports_vision  INTEGER,          -- NULL = auto-detect
    is_default  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL,
    PRIMARY KEY (owner, name)
);
CREATE INDEX IF NOT EXISTS idx_providers_owner ON provider_profiles (owner);

CREATE TABLE IF NOT EXISTS settings (
    owner      TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,          -- JSON
    updated_at REAL NOT NULL,
    PRIMARY KEY (owner, key)
);

-- Environment values a workflow or an MCP server needs: connection strings,
-- database passwords, API keys for a third-party server. Separate from `settings`
-- because these are not preferences — they are read by machinery at connect time
-- and at tool-call time, they are never sent to a model, and they are write-only
-- over the API. A preference that leaks is untidy; one of these that leaks is an
-- incident.
CREATE TABLE IF NOT EXISTS secrets (
    owner       TEXT NOT NULL,
    name        TEXT NOT NULL,
    value       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    -- Which tool this credential is *for* (`sql`, `http`), so a node configuring
    -- that tool can offer the ones you already made instead of asking you to
    -- remember a variable name. Empty means general-purpose — every secret that
    -- existed before this column, and the ones an MCP server reads from its env.
    --
    -- Deliberately not a *node* id. A credential belongs to a kind of service,
    -- not to a position in one graph: keying it by node would mean copying the
    -- value to reuse it and editing every copy to rotate it. The node-specific
    -- part is the binding — node A names PROD_DB, node B names STAGING_DB —
    -- which lives on the node and always has.
    tool        TEXT NOT NULL DEFAULT '',
    updated_at  REAL NOT NULL,
    PRIMARY KEY (owner, name)
);
CREATE INDEX IF NOT EXISTS idx_secrets_owner ON secrets (owner);

-- One row per completed one-off migration, so importing providers.json cannot
-- run twice and resurrect profiles the user has since deleted.
CREATE TABLE IF NOT EXISTS migrations (
    id         TEXT PRIMARY KEY,
    applied_at REAL NOT NULL
);
"""


@dataclass(frozen=True)
class ProviderRow:
    """A stored profile. ``api_key`` stays server-side; :meth:`public` masks it."""

    owner: str
    name: str
    kind: str
    base_url: str | None
    model: str
    api_key: str | None
    context_window: int
    max_output_tokens: int
    supports_vision: bool | None
    is_default: bool
    created_at: float

    def profile(self) -> ProviderProfile:
        """The library-level profile the provider factory understands."""
        return ProviderProfile(
            name=self.name,
            kind=self.kind,  # type: ignore[arg-type]
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            supports_vision=self.supports_vision,
        )

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "endpoint": self.profile().endpoint(),
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_vision": self.supports_vision,
            "is_default": self.is_default,
            "created_at": self.created_at,
            # Enough to recognise which key is set, not enough to use it.
            "api_key_masked": mask_secret(self.api_key),
            "has_api_key": bool(self.api_key),
        }


class SettingsStore:
    """Provider profiles + UI preferences. Thread-safe: uvicorn serves from a pool."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from neurosurfer.config.paths import artifacts_home

            path = artifacts_home() / "settings.db"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._add_missing_columns()
            self._db.commit()

    #: Columns added to tables that already exist in the wild. `CREATE TABLE IF
    #: NOT EXISTS` is a no-op against a database created before the column was
    #: written, so a new field would be present on a fresh install and missing on
    #: every real one — the sort of difference that only shows up on someone
    #: else's machine.
    _ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
        ("secrets", "tool", "TEXT NOT NULL DEFAULT ''"),
    )

    def _add_missing_columns(self) -> None:
        for table, column, decl in self._ADDED_COLUMNS:
            have = {
                r["name"]
                for r in self._db.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in have:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── provider profiles ───────────────────────────────────────────────────
    @staticmethod
    def _row(r: sqlite3.Row) -> ProviderRow:
        return ProviderRow(
            owner=r["owner"], name=r["name"], kind=r["kind"], base_url=r["base_url"],
            model=r["model"], api_key=r["api_key"],
            context_window=r["context_window"],
            max_output_tokens=r["max_output_tokens"],
            supports_vision=None if r["supports_vision"] is None else bool(r["supports_vision"]),
            is_default=bool(r["is_default"]), created_at=r["created_at"],
        )

    def list_providers(self, owner: str = DEFAULT_OWNER) -> list[ProviderRow]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM provider_profiles WHERE owner = ? ORDER BY created_at",
                (owner,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def get_provider(self, name: str, owner: str = DEFAULT_OWNER) -> ProviderRow | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM provider_profiles WHERE owner = ? AND name = ?",
                (owner, name),
            ).fetchone()
        return self._row(row) if row else None

    def default_provider(self, owner: str = DEFAULT_OWNER) -> ProviderRow | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM provider_profiles WHERE owner = ? AND is_default = 1",
                (owner,),
            ).fetchone()
        return self._row(row) if row else None

    def add_provider(self, profile: ProviderProfile, *, owner: str = DEFAULT_OWNER,
                     make_default: bool | None = None) -> ProviderRow:
        """Insert a profile. Raises :class:`ValueError` if the name is taken."""
        now = time.time()
        with self._lock:
            existing = self._db.execute(
                "SELECT COUNT(*) AS n FROM provider_profiles WHERE owner = ?", (owner,)
            ).fetchone()["n"]
            # The first profile is the default whether asked for or not — otherwise a
            # user configures one provider and nothing resolves to it.
            is_default = existing == 0 if make_default is None else bool(make_default)
            try:
                self._db.execute(
                    "INSERT INTO provider_profiles (owner, name, kind, base_url, model,"
                    " api_key, context_window, max_output_tokens, supports_vision,"
                    " is_default, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (owner, profile.name, profile.kind, profile.base_url, profile.model,
                     profile.api_key, profile.context_window, profile.max_output_tokens,
                     None if profile.supports_vision is None else int(profile.supports_vision),
                     int(is_default), now),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"A provider named '{profile.name}' already exists.") from e
            if is_default:
                self._clear_other_defaults(owner, profile.name)
            self._db.commit()
        row = self.get_provider(profile.name, owner)
        assert row is not None
        return row

    def update_provider(self, name: str, changes: dict[str, Any],
                        *, owner: str = DEFAULT_OWNER) -> ProviderRow:
        """Patch the given fields. Absent keys are left alone; ``api_key`` is only
        replaced when a new one is supplied, so a form that never received the key
        cannot blank it by echoing back what it was shown."""
        allowed = {"kind", "base_url", "model", "api_key", "context_window",
                   "max_output_tokens", "supports_vision"}
        sets, values = [], []
        for key, value in changes.items():
            if key not in allowed or value is None:
                continue
            if key == "supports_vision":
                value = int(bool(value))
            sets.append(f"{key} = ?")
            values.append(value)
        with self._lock:
            if self.get_provider(name, owner) is None:
                raise KeyError(f"No provider named '{name}'.")
            if sets:
                self._db.execute(
                    f"UPDATE provider_profiles SET {', '.join(sets)}"
                    " WHERE owner = ? AND name = ?",
                    (*values, owner, name),
                )
                self._db.commit()
        row = self.get_provider(name, owner)
        assert row is not None
        return row

    def delete_provider(self, name: str, owner: str = DEFAULT_OWNER) -> None:
        with self._lock:
            row = self.get_provider(name, owner)
            if row is None:
                raise KeyError(f"No provider named '{name}'.")
            self._db.execute(
                "DELETE FROM provider_profiles WHERE owner = ? AND name = ?", (owner, name)
            )
            # Deleting the default promotes the oldest survivor, so a workspace with
            # providers always has one that plain `{}` resolves to.
            if row.is_default:
                nxt = self._db.execute(
                    "SELECT name FROM provider_profiles WHERE owner = ?"
                    " ORDER BY created_at LIMIT 1", (owner,),
                ).fetchone()
                if nxt is not None:
                    self._db.execute(
                        "UPDATE provider_profiles SET is_default = 1"
                        " WHERE owner = ? AND name = ?", (owner, nxt["name"]),
                    )
            self._db.commit()

    def set_default_provider(self, name: str, owner: str = DEFAULT_OWNER) -> None:
        with self._lock:
            if self.get_provider(name, owner) is None:
                raise KeyError(f"No provider named '{name}'.")
            self._db.execute(
                "UPDATE provider_profiles SET is_default = 1 WHERE owner = ? AND name = ?",
                (owner, name),
            )
            self._clear_other_defaults(owner, name)
            self._db.commit()

    def _clear_other_defaults(self, owner: str, keep: str) -> None:
        self._db.execute(
            "UPDATE provider_profiles SET is_default = 0 WHERE owner = ? AND name != ?",
            (owner, keep),
        )

    # ── key/value settings ──────────────────────────────────────────────────
    def get_settings(self, owner: str = DEFAULT_OWNER) -> dict[str, Any]:
        with self._lock:
            rows = self._db.execute(
                "SELECT key, value FROM settings WHERE owner = ?", (owner,)
            ).fetchall()
        out: dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except json.JSONDecodeError:
                continue  # a corrupt row is a missing preference, not an error
        return out

    def put_settings(self, values: dict[str, Any], owner: str = DEFAULT_OWNER) -> dict[str, Any]:
        """Merge *values* into the owner's settings. A ``None`` clears its key."""
        now = time.time()
        with self._lock:
            for key, value in values.items():
                if value is None:
                    self._db.execute(
                        "DELETE FROM settings WHERE owner = ? AND key = ?", (owner, key)
                    )
                    continue
                self._db.execute(
                    "INSERT INTO settings (owner, key, value, updated_at) VALUES (?,?,?,?)"
                    " ON CONFLICT (owner, key) DO UPDATE SET value = excluded.value,"
                    " updated_at = excluded.updated_at",
                    (owner, key, json.dumps(value), now),
                )
            self._db.commit()
        return self.get_settings(owner)

    # ── secrets ─────────────────────────────────────────────────────────────
    def list_secrets(
        self, owner: str = DEFAULT_OWNER, *, tool: str | None = None
    ) -> list[dict[str, Any]]:
        """Every secret's *metadata* — never a value.

        Deliberately not a `get_secrets`-with-masking: the only caller that wants
        values is machinery resolving `${VAR}`, and it should have to say so by
        name. A listing that quietly carries plaintext is one careless log line
        away from being the leak.

        *tool* narrows to the credentials made for one kind of tool, which is
        what a node's credential picker asks for: configuring a second `sql` node
        should offer the connection you set up for the first, not the whole store.
        Untagged secrets are always included — they predate the tag and may well
        be the connection you want.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT name, value, description, tool, updated_at FROM secrets"
                " WHERE owner = ? ORDER BY name",
                (owner,),
            ).fetchall()
        wanted = str(tool or "").strip()
        return [
            {
                "name": r["name"],
                "description": r["description"],
                "tool": r["tool"],
                "updated_at": r["updated_at"],
                "masked": mask_secret(r["value"]),
            }
            for r in rows
            if not wanted or r["tool"] in ("", wanted)
        ]

    def secret_values(self, owner: str = DEFAULT_OWNER) -> dict[str, str]:
        """Every secret as ``{name: value}`` — for `${VAR}` resolution only."""
        with self._lock:
            rows = self._db.execute(
                "SELECT name, value FROM secrets WHERE owner = ?", (owner,)
            ).fetchall()
        return {r["name"]: r["value"] for r in rows}

    def put_secret(self, name: str, value: str, *, description: str = "",
                   tool: str = "", owner: str = DEFAULT_OWNER) -> None:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO secrets (owner, name, value, description, tool, updated_at)"
                " VALUES (?,?,?,?,?,?)"
                " ON CONFLICT (owner, name) DO UPDATE SET value = excluded.value,"
                " description = excluded.description, tool = excluded.tool,"
                " updated_at = excluded.updated_at",
                (owner, name, value, description, tool, now),
            )
            self._db.commit()

    def delete_secret(self, name: str, owner: str = DEFAULT_OWNER) -> bool:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM secrets WHERE owner = ? AND name = ?", (owner, name)
            )
            self._db.commit()
        return cur.rowcount > 0

    # ── one-off import of the library's JSON store ──────────────────────────
    def import_json_profiles(self, *, owner: str = DEFAULT_OWNER,
                             store: Any = None) -> list[str]:
        """Copy ``~/.neurosurfer/providers.json`` into *owner*, once, ever.

        Recorded in ``migrations`` rather than inferred from the table being empty:
        "no profiles" is also what a user who deleted them all looks like, and
        re-importing would resurrect what they removed.
        """
        marker = f"import_json_profiles:{owner}"
        with self._lock:
            done = self._db.execute(
                "SELECT 1 FROM migrations WHERE id = ?", (marker,)
            ).fetchone()
            if done:
                return []

        if store is None:
            from neurosurfer.config.profiles import ProviderStore

            store = ProviderStore.default()

        imported: list[str] = []
        active = None
        try:
            profiles = store.list()
            active = store.active_name()
        except Exception:  # noqa: BLE001 - an unreadable legacy file must not block boot
            profiles = []
        for profile in profiles:
            try:
                self.add_provider(profile, owner=owner,
                                  make_default=(profile.name == active))
                imported.append(profile.name)
            except ValueError:
                continue  # already present under this owner — leave it alone

        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO migrations (id, applied_at) VALUES (?, ?)",
                (marker, time.time()),
            )
            self._db.commit()
        return imported
