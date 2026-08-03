"""The registry's icon set — a tool's face, resolved rather than guessed at.

A palette that lists `sql`, `http`, `apply_edit` in a column of identical rows is
showing the engine's vocabulary to somebody who never agreed to learn it. An icon
is the cheapest way to say *database* / *web* / *this touches your files* before
any word is read, and the canvas needs one per node for the same reason.

**The icons live here, next to the tools they belong to, and are served from
here.** The alternative — a sprite sheet in the front-end keyed by tool name —
puts the artwork in a codebase that has no idea what tools exist, so every
imported MCP server and every authored tool arrives faceless and stays that way
until somebody edits the studio. The registry already knows what a tool is; it
should know what it looks like too.

**Three tiers, most specific first**, which is what makes per-tool artwork
optional instead of a prerequisite:

1. what the tool *declares* (``Tool.icon``), for artwork that carries a product's
   own identity;
2. an icon named after the tool, so adding ``icons/browse.svg`` is the whole act
   of giving ``browse`` an icon — no registration step to forget;
3. the **family** default, from which registry domain the tool lives in. This is
   the tier that matters: an MCP server nobody has ever seen still reads as
   *database* rather than as a generic box.

Every icon is a 24×24 stroke drawing using ``currentColor``, so one file works on
a dark canvas, a light palette row and a disabled state without variants.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["DOMAIN_ICONS", "FALLBACK_ICON", "icon_bytes", "icon_slugs", "resolve_icon"]

ICON_DIR = Path(__file__).parent / "icons"

#: Registry domain → the icon every tool in it falls back to. The domains are the
#: directories under `registry/core/`, so this list is a mirror of the package
#: layout and a new domain shows up as a missing key rather than a wrong picture.
DOMAIN_ICONS: dict[str, str] = {
    "agent": "agent",
    "data": "data",
    "database": "database",
    "filesystem": "filesystem",
    "system": "system",
    "web": "web",
}

#: Origins that are not registry domains at all, but still want a family face.
ORIGIN_ICONS: dict[str, str] = {
    "mcp": "mcp",
    "imported": "mcp",
    "authored": "authored",
}

#: Used when nothing else resolves. Present as a file, so the API never has to
#: answer a 404 for an icon and no front-end needs a broken-image state.
FALLBACK_ICON = "tool"


def _domain_of(tool: Any) -> str:
    """Which `registry/core/<domain>/` package a tool's class was defined in.

    Derived from the module path rather than declared, because the directory
    layout is already the grouping — asking each tool to also *say* which folder
    it is in creates a second fact that can disagree with the first.
    """
    module = str(getattr(type(tool), "__module__", "") or "")
    marker = "neurosurfer.registry.core."
    if marker not in module:
        return ""
    return module.split(marker, 1)[1].split(".", 1)[0]


@lru_cache(maxsize=1)
def icon_slugs() -> frozenset[str]:
    """Every icon the registry ships, by slug.

    Cached: it is read on every catalog build and the set only changes when the
    package on disk does.
    """
    if not ICON_DIR.is_dir():
        return frozenset()
    return frozenset(p.stem for p in ICON_DIR.glob("*.svg"))


def resolve_icon(tool: Any) -> str:
    """The icon slug for *tool* — always a slug that exists on disk.

    Never returns an empty string or a name with no file behind it: a caller
    rendering this should not have to hold a fallback of its own, and the whole
    point of resolving here is that the answer is usable.
    """
    have = icon_slugs()

    declared = str(getattr(tool, "icon", "") or "").strip()
    if declared in have:
        return declared

    name = str(getattr(tool, "name", "") or "").strip()
    if name in have:
        return name

    domain = DOMAIN_ICONS.get(_domain_of(tool), "")
    if domain in have:
        return domain

    origin = "mcp" if getattr(tool, "is_mcp", False) else str(getattr(tool, "origin", ""))
    family = ORIGIN_ICONS.get(origin, "")
    if family in have:
        return family

    return FALLBACK_ICON


def icon_bytes(slug: str) -> bytes | None:
    """The SVG for *slug*, or None if there is no such icon.

    The slug is matched against the known set rather than joined onto a path:
    this is reached from an HTTP route, and `../../etc/passwd` is a path that
    resolves perfectly well.
    """
    clean = str(slug or "").strip().removesuffix(".svg")
    if clean not in icon_slugs():
        return None
    try:
        return (ICON_DIR / f"{clean}.svg").read_bytes()
    except OSError:
        return None
