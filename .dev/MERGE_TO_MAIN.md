# Merging to `main` — the plan

**Nothing here has been executed.** This is the procedure, ready to run.

`main` is at `4065c2f`. The branch is `architect-validator/enhancement`, pushed,
88 commits ahead.

---

## 1. What `main` gets, and what it does not

The owner's call: **`main` is the published project, the branch is the workshop.**

| Path | To `main`? | Why |
|---|---|---|
| `neurosurfer/` | ✅ all | the package |
| `docs/`, `README.md`, `CHANGELOG.md` | ✅ all | the published record |
| `tutorials/` | ✅ all | 7 notebooks, the on-ramp |
| `pyproject.toml`, `LICENSE`, `.github/` | ✅ | build, licence, CI |
| **`.dev/`** | ❌ **none** | plans, build logs, handoffs, briefs — working material |
| **`tests/`** | ⚠️ **six modules** | see §2 |

### The concern, recorded once

Removing the suite has a cost worth naming, so that choosing it stays a choice:

- **The conformance suite is the claim.** "Implements `BaseVectorDB`" currently
  *means* "passes `tests/vectorstores/conformance.py`". Without it shipping, that
  sentence in the README is an assertion rather than a check anyone can run.
- **Contributors cannot verify a PR.** CI runs `pytest -q`, so it will keep
  passing — over six files instead of ninety-nine. A regression in the other
  ninety-three lands green.
- **The evidence for this release's bug fixes leaves with them.** The `title`
  schema bug, the OTLP regression, the blocking-validation rule — each shipped
  with a test that fails without the fix.

Kept in mind while choosing the six below: they are picked to be the *most* of
that value in the least code.

---

## 2. The six that stay

206 tests, **3.1 seconds**, no network, no API key. Fast enough that a
contributor will actually run them.

| Module | What it holds the line on |
|---|---|
| `tests/test_agent_loop.py` | the agent loop — turns, tool calls, guardrails, streaming events |
| `tests/test_provider_parity.py` | every provider behaves the same behind one protocol |
| `tests/test_workflow_package.py` | packages load, validate and register |
| `tests/engine/test_graph_control_flow.py` | the DAG engine — router, loop, map, subgraph |
| `tests/engine/test_validation_rules.py` | the gate: what blocks a graph and what only warns |
| `tests/vectorstores/` | the conformance suite — the vector-store contract itself |

Plus the four support files they need, which are not tests:
`tests/__init__.py`, `tests/conftest.py`, `tests/fakes.py`,
`tests/engine/__init__.py`.

Verified as a standalone set:

```
pytest -q tests/test_agent_loop.py tests/test_provider_parity.py \
          tests/test_workflow_package.py tests/engine/test_graph_control_flow.py \
          tests/engine/test_validation_rules.py tests/vectorstores
→ 206 passed in 3.12s
```

---

## 3. Squash, not a merge commit

A normal merge puts all 88 commits into `main`'s history — and with them every
`.dev/` file and all ninety-nine test modules, retrievable by anyone who runs
`git log --all`. Deleting them in a follow-up commit changes the *tree*, not the
history.

So: **squash**. `main` gets one release commit containing the final tree; the
branch keeps the full history, which is where it is useful.

```bash
git checkout main
git merge --squash architect-validator/enhancement
# the index now holds the branch's whole tree, uncommitted:
git rm -r --cached .dev
rm -rf .dev
bash <<'TRIM'
keep="tests/__init__.py tests/conftest.py tests/fakes.py tests/engine/__init__.py
      tests/test_agent_loop.py tests/test_provider_parity.py tests/test_workflow_package.py
      tests/engine/test_graph_control_flow.py tests/engine/test_validation_rules.py"
for f in $(git ls-files tests); do
  case " $keep " in *" $f "*) continue;; esac
  case "$f" in tests/vectorstores/*) continue;; esac
  git rm -q --cached "$f"; rm -f "$f"
done
find tests -type d -empty -delete
TRIM
```

Then, before committing:

```bash
pytest -q                     # expect 206 passed
ruff check neurosurfer tests  # CI runs exactly this
```

---

## 4. The version bump — part of the same commit

`1.0.0` → **`2.0.0`**, in three places that must move together:

- `pyproject.toml` — `version = "2.0.0"`
- `neurosurfer/__init__.py` — `__version__ = "2.0.0"`
- `CHANGELOG.md` — `## [Unreleased]` → `## [2.0.0] — <date>`

Major, not minor, because four things break on upgrade: the `tools.builtin`
submodule paths, `neurosurfer.llm.pricing`, four public methods/fields, and a
validation warning promoted to an error that stops a workflow already on disk.
The CHANGELOG's *Upgrading from 1.0.0* table is the migration note.

---

## 5. Add `.dev/` to `.gitignore` on `main`

So it cannot come back by accident on a later merge:

```
.dev/
```

Do **not** add it on the branch — the branch is where it lives.

---

## 6. Order of operations

1. `git checkout main && git merge --squash architect-validator/enhancement`
2. Remove `.dev/`; trim `tests/` to the six
3. Add `.dev/` to `.gitignore`
4. Bump the version in all three places
5. `pytest -q` → 206 · `ruff check neurosurfer tests` → clean
6. `mkdocs build --strict` → clean
7. Commit, `git push origin main`
8. Tag `v2.0.0`
9. The branch is untouched and keeps everything

---

## 7. Two things that had to be fixed first, and were

- **Two shipping files pointed into `.dev/`** — a docstring in
  `graph/engine/json_schema.py` citing a plan file, and a comment in
  `engine/test_validation_rules.py` (one of the six that stay). Both would have
  been dangling references the moment `.dev/` stopped shipping. Rewritten to say
  the thing rather than point at it.
- **The `title` schema fix was missing from the CHANGELOG** — the most
  user-visible bug fixed this session, and the record of it lives in the
  CHANGELOG once `.dev/` is gone. Added.
