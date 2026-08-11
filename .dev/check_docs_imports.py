"""Every `neurosurfer` import in `docs/` resolves against the installed package.

The cheap half of "do the samples work". It does not execute a sample — most need
a provider and an API key — but it does prove that every module a page names
exists and exports every name the page imports from it. That is the failure mode
docs actually have: a symbol renamed in the source and left behind in the prose.

It found `build_provider_from_profile` in `server/agents.md`, which has never
existed; the function is `build_provider` and it takes a `Config`.

Run from the repo root, in an environment with the package installed:

    PYTHONIOENCODING=utf-8 python .dev/check_docs_imports.py

`PYTHONIOENCODING` is not optional on Windows: importing `neurosurfer` prints a
banner containing box-drawing characters, and the default cp1252 stdout cannot
encode them, so the import dies before anything is checked. Same root cause as
the subprocess failures in WINDOWS_TEST_FAILURES.md.

Two things are deliberately tolerated:

* **A module needing an extra that is not installed** is reported separately and
  does not fail the run — `neurosurfer.rag` without `chromadb` says nothing about
  the docs.
* **An import a page documents as broken** is expected to fail. `upgrading.md`
  shows `from neurosurfer.tools.builtin.search import SearchTool` precisely to say
  it no longer works, so that failing is the page being correct. Such lines are
  marked in the docs with a trailing `# ❌` comment and inverted here.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
_FROM = re.compile(r"^from ([\w.]+) import ([^\n(]+)$", re.M)
_IMPORT = re.compile(r"^import ([\w.]+)\s*(#.*)?$", re.M)

#: An extra we do not install for this check; a failure naming one is not a docs bug.
OPTIONAL = ("chromadb", "sentence_transformers", "playwright", "langfuse",
            "opentelemetry", "mcp", "ddgs", "tiktoken", "torch", "fitz")


def _strip_comment(text: str) -> tuple[str, bool]:
    """Drop a trailing `# …`; report whether it marked the line as broken-on-purpose."""
    if "#" not in text:
        return text.strip(), False
    code, comment = text.split("#", 1)
    return code.strip(), "❌" in comment


def _collect() -> dict[tuple[str, str], tuple[list[str], bool]]:
    found: dict[tuple[str, str], tuple[list[str], bool]] = {}
    for page in sorted(DOCS.rglob("*.md")):
        for block in _BLOCK.findall(page.read_text(encoding="utf-8")):
            for mod, names in _FROM.findall(block):
                if not mod.startswith("neurosurfer"):
                    continue
                clean, expect_fail = _strip_comment(names)
                key = (mod, clean)
                pages, prior = found.get(key, ([], False))
                found[key] = ([*pages, page.relative_to(ROOT).as_posix()], prior or expect_fail)
            for mod, comment in _IMPORT.findall(block):
                if not mod.startswith("neurosurfer"):
                    continue
                key = (mod, "")
                pages, prior = found.get(key, ([], False))
                found[key] = ([*pages, page.relative_to(ROOT).as_posix()],
                              prior or "❌" in (comment or ""))
    return found


def main() -> int:
    quiet = io.StringIO()
    problems: list[str] = []
    skipped: list[str] = []
    checked = 0

    for (mod, names), (pages, expect_fail) in sorted(_collect().items()):
        checked += 1
        where = pages[0]
        try:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                module = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else ""
            if expect_fail:
                continue                                  # the page says so
            if missing.startswith(OPTIONAL) or any(o in missing for o in OPTIONAL):
                skipped.append(f"{mod} (needs {missing})")
                continue
            problems.append(f"module  {mod}  <- {where}  ({e})")
            continue
        except Exception as e:  # noqa: BLE001 - any import failure is a finding
            problems.append(f"module  {mod}  <- {where}  ({type(e).__name__}: {e})")
            continue

        if expect_fail:
            problems.append(f"expected-broken import resolved: {mod} <- {where}")
            continue

        for raw in names.split(","):
            name = raw.strip().split(" as ")[0].strip()
            if not name:
                continue
            try:
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    ok = hasattr(module, name)
            except Exception:  # noqa: BLE001 - lazy attr pulling an optional dep
                skipped.append(f"{mod}.{name} (lazy optional dep)")
                continue
            if not ok:
                problems.append(f"name    {mod}.{name}  <- {where}")

    for p in problems:
        print(p)
    if skipped:
        print(f"\nskipped, optional deps not installed: {len(skipped)}")
    print(f"\n{checked} distinct imports checked, {len(problems)} problem(s).")
    return len(problems)


if __name__ == "__main__":
    sys.exit(main())
