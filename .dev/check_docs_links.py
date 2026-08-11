"""Static link, anchor and nav integrity check over `docs/`.

Exists because `mkdocs build --strict` needs the docs toolchain installed, and the
checks that catch the failures we actually made — a link to a page that moved, an
anchor to a section that was split out, a page left out of the nav — need nothing
but the tree. Run it before pushing docs; run `mkdocs build --strict` as well when
you have an environment.

    python .dev/check_docs_links.py        # from the repo root

Exit code is the number of problems, so it drops into CI unchanged.

The slug function mirrors `markdown.extensions.toc.slugify` exactly, including the
collapse of runs of spaces and hyphens into one. A naive version that skips the
collapse reports every heading containing an em dash as a broken anchor, which is
four false positives on this repo's docs and enough noise to make the check
ignorable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"

_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$")
_EXPLICIT_ANCHOR = re.compile(r"\{\s*#([\w-]+)\s*\}")
_LINK = re.compile(r"\]\(([^)\s]+)")
_ANCHORED = re.compile(r"\]\(([^)\s]*)#([\w-]+)\)")


def slugify(text: str) -> str:
    """`markdown.extensions.toc.slugify`, separator `-`."""
    value = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def anchors(path: Path) -> set[str]:
    """Every anchor a page defines — derived slugs plus `{ #explicit }` ones."""
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _HEADING.match(line)
        if not m:
            continue
        title = m.group(1)
        if explicit := _EXPLICIT_ANCHOR.search(title):
            found.add(explicit.group(1))
            title = _EXPLICIT_ANCHOR.sub("", title)
        found.add(slugify(title))
    return found


def main() -> int:
    problems: list[str] = []
    pages = sorted(DOCS.rglob("*.md"))

    for page in pages:
        lines = page.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines, 1):
            for m in _LINK.finditer(line):
                target = m.group(1)
                if target.startswith(("http", "mailto:")):
                    continue
                rel = target.split("#")[0]
                if rel and not (page.parent / rel).resolve().exists():
                    problems.append(f"missing file   {page.relative_to(ROOT)}:{n} -> {target}")
            for m in _ANCHORED.finditer(line):
                target, anchor = m.group(1), m.group(2)
                if target.startswith("http"):
                    continue
                dest = (page.parent / target).resolve() if target else page
                if dest.exists() and dest.suffix == ".md" and anchor not in anchors(dest):
                    problems.append(f"missing anchor {page.relative_to(ROOT)}:{n} -> {target}#{anchor}")

    nav = MKDOCS.read_text(encoding="utf-8").split("nav:")[1].split("extra_css:")[0]
    listed = re.findall(r"([\w./-]+\.md)", nav)
    problems += [f"nav -> missing  {t}" for t in listed if not (DOCS / t).exists()]
    problems += [
        f"not in nav      {p.relative_to(DOCS).as_posix()}"
        for p in pages
        if p.relative_to(DOCS).as_posix() not in listed
    ]

    for p in problems:
        print(p)
    print(f"\n{len(pages)} pages checked, {len(problems)} problem(s).")
    return len(problems)


if __name__ == "__main__":
    sys.exit(main())
