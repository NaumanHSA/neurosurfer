# The suite on Windows — 16 failures, and ten are the library, not the tests

> **Deferred, deliberately — 2026-08-16.** This release is verified on **Linux**
> and Windows is not being re-run for it. The ten failures below still name real
> tests and were last measured ~50 commits ago, so treat the numbers as indicative
> rather than current. Seven are one upstream bug (subprocess output decoded with
> the ANSI codepage) and three are `os.killpg` called unconditionally; neither is
> this branch's work, and neither has been re-checked since.
>
> `pyproject.toml` still declares `Operating System :: OS Independent`. That claim
> is now ahead of the evidence — either re-run this suite before it goes out, or
> say plainly in the release notes which platform was tested.

**As of 2026-08-10, branch `architect-validator/enhancement` at `1fbbadb`.**
`NEUROSURFER_TEST_BASE_URL=http://127.0.0.1:9 python -m pytest tests/ -q -p no:randomly`
→ **1207 passed, 16 failed, 4 skipped**, ~60s.

None of them are new. They fail identically before and after every commit on
this branch — verified by stashing the working tree and re-running, which is the
check worth repeating rather than trusting this file.

## This supersedes the build log's *Running the suite on Windows*

§9 did open these, and its diagnosis was correct when written: thirteen
`tests/tools/` failures because fixtures hardcode `ToolContext(cwd=Path("/tmp"))`,
which on Windows is a **drive-relative** path, and launching a subprocess in a
directory that does not exist gives `[WinError 267] The directory name is
invalid`.

That is no longer what happens, and the reason is worth more than the finding:

```
Path("/tmp") resolves to : D:\tmp
exists                   : True
```

**`D:\tmp` has since been created on this machine**, so the fixtures now land
somewhere real and those thirteen failures resolved into a different and smaller
set with entirely different causes. `WinError 267` appears zero times in the
current run.

So the suite's Windows behaviour depends on whether a directory nobody declared
happens to exist on the drive the repo is checked out on. That is worth knowing
before reading either diagnosis as fixed, and it is a sharper version of §9's own
advice — *compare failure lists, not counts*. The count went 15 → 16 while
almost none of the members stayed the same.

What follows is the current list, read one at a time. The part §9 could not
reach, because the `/tmp` failures were masking it, is that **ten of the sixteen
are library bugs rather than test-environment noise.**

---

## What they are

| # | Group | Where the defect is |
|---|---|---|
| 7 | A — Subprocess output decoded with the ANSI codepage | **library** |
| 5 | B — Path separators in test assertions | tests |
| 3 | C — POSIX-only APIs called unconditionally | **library** |
| 1 | D — POSIX file modes | tests assert it; the *intent* is unmet on Windows |

---

## A — Subprocess output is decoded with the ANSI codepage (7 failures)

**This is a library bug and it is not confined to the suite.**

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 296
```

Eleven call sites pass `text=True` to `subprocess` without `encoding=`, so
Python decodes the child's output with `locale.getpreferredencoding()` — UTF-8
on Linux, **cp1252 on a default Windows install**:

```
neurosurfer/app/cli/commands/pyenv.py:95
neurosurfer/architect/agent/verify.py:512
neurosurfer/architect/tool_author.py:327
neurosurfer/mcp/manager.py:90
neurosurfer/prompts/environment.py:25, :33
neurosurfer/registry/core/system/python_exec/interpreter.py:89
neurosurfer/registry/core/system/python_exec/managed_env.py:94, :103, :122
tests/engine/test_import_boundaries.py:29
```

What trips it is our own startup banner: importing `neurosurfer` prints box
characters (`█`, `╗`, `─`) that cp1252 cannot decode, so *any* child process
that imports the package returns undecodable bytes. The failure surfaces far
from the cause — as `'NoneType' object has no attribute 'strip'`, or a missing
key in a results dict — because the caller only sees that it got nothing back.

Failing because of it:

- `tests/engine/test_import_boundaries.py` (2) — the guard `HANDOFF.md` §2 says
  must never be deleted. **It is currently not guarding anything on Windows**:
  it cannot read the subprocess it inspects, so it fails for a reason unrelated
  to import boundaries and would fail the same way if a boundary were genuinely
  broken. On Windows it is noise, not a net.
- `tests/tools/test_tool_author.py` (3) — `KeyError: 'call_is_async'`; the
  sandbox's verification subprocess is unreadable, so its checks never arrive.
- `tests/architect/test_workflow_architect.py::TestFunctionalSandbox` (2) —
  `assert None is True`, same sandbox.

**User-facing consequence:** on Windows the tool author cannot verify a tool it
wrote, and the Architect's functional sandbox cannot report. Neither is a test
artefact. The fix is `encoding="utf-8", errors="replace"` at each site, and it
is small — it is listed here rather than done because it is unrelated to the
work on this branch and deserves its own commit and its own regression test.

## B — Path separators in test assertions (5 failures)

Test-side assumptions about `/`, harmless to users:

- `tests/tools/test_tools.py::test_list_dir_and_glob` —
  `assert 'sub/y.py' in 'sub\y.py\nx.py'`
- `tests/tools/test_python_env_management.py::test_managed_spec_provisions` —
  `'\managed\bin\python' == '/managed/bin/python'`
- `tests/tools/test_python_env_management.py::test_conda_spec_found` — same shape
- `tests/tools/test_tool_enhancements.py::test_cwd_override` — the test expects
  `C:\Users\...\sub`; bash's `pwd` reports `/tmp/pytest-of-Pc/...`
- `tests/test_agent_loop.py::test_write_outside_scope_always_widens_and_persists`
  — `any(str(out) in s for s in g.write_scope)` compares unnormalised paths

Worth fixing with `Path`/`os.sep` comparisons rather than string containment, at
which point the assertions get stronger on Linux too.

## C — POSIX-only APIs called unconditionally (3 failures)

**Also a library bug.**

```
AttributeError: module 'os' has no attribute 'killpg'
  neurosurfer/registry/core/system/python_exec/sandbox.py:225
```

`os.killpg` does not exist on Windows, so the **timeout and kill paths of
`python_exec` and `run_command` cannot work there at all** — a runaway child is
not killable, and the timeout returns an `AttributeError` instead of a timeout.

- `tests/tools/test_python_exec.py::TestTimeout::test_timeout_returns_error`
- `tests/tools/test_tool_enhancements.py::TestRunCommandCwdAndBackground::test_kill_background_job`

And a third, the same shape one layer up:

- `tests/tools/test_python_env_management.py::TestResolveEnvSpec::test_path_spec_to_venv_dir`
  — `interpreter.py:139` looks for `bin/python`; a Windows venv puts it at
  `Scripts/python.exe`, so no interpreter is ever found under a venv directory.

## D — POSIX file modes (1 failure)

- `tests/test_cli.py::test_file_permissions_are_owner_only` — `assert 438 == 384`,
  i.e. `0o666` vs the expected `0o600`. `chmod` does not carry group/other bits
  on Windows. The *intent* — a credentials file readable only by its owner — is
  not achievable this way on Windows and would need an ACL. Today the file is
  simply not restricted, which is worth knowing before anyone relies on it.

---

## Intermittent, and not in the sixteen

Six subprocess tests in `tests/tools/test_python_exec.py`
(`TestResultVariable` ×3, `TestStderrAndExit` ×2, `TestEnvSanitisation` ×1)
failed on three consecutive runs and passed on the next two, with no code change
between them. Treat a run of 16 or 22 failures as the same result. If you are
bisecting, pin on the named tests rather than on the count.

---

## What to do with this

Nothing blocks the merge to `main` — none of it was introduced on this branch.
Three entries belong on the roadmap as their own work, in this order:

1. **`encoding="utf-8"` at the eleven subprocess sites** (group A). Smallest fix,
   largest user-facing effect, and it restores the import-boundary guard that
   `HANDOFF.md` §2 says must never be lost — which is currently lost on Windows
   without anything saying so.
2. **A Windows path for process termination and venv layout** (group C).
   `os.killpg` is not a portability nicety here: it is the only thing that ends
   a runaway child.
3. **Stop hardcoding `/tmp` in fixtures.** `tmp_path` exists. Until then the
   suite's result on Windows depends on whether `D:\tmp` happens to exist, which
   is how the same branch produced two different failure sets a week apart.

Re-run and re-read before trusting any of the above; the last two diagnoses of
this list both aged badly, this one included.
