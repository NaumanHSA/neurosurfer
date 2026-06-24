# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.2.x (latest) | Yes |
| 0.1.x | No — please upgrade |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email the maintainer directly at **naumanhsa965@gmail.com** with:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (including any relevant config or Task YAML).
- Any proof-of-concept code.

You will receive an acknowledgement within **48 hours** and a resolution
timeline within **7 days**. We will coordinate a disclosure date with you
before publishing any fix.

## Scope

The following are in scope for security reports:

- Guardrail bypass — a Task or external input that causes the agent to write
  outside its declared `write_scope` or execute commands beyond its `shell_policy`.
- Policy ceiling bypass — authoring a Task that exceeds the hard caps enforced
  by `PolicyCeiling` (e.g. ungated shell, write scope including `/` or `~`).
- Path traversal in `read_file`, `list_dir`, or `write_file` tools.
- API key or secret exposure in logs, transcripts, or error messages.
- Prompt injection via tool results that causes the agent to take actions
  outside the scope of the active Task.

Out of scope: issues in third-party dependencies (report those upstream),
denial-of-service against a local model server, and issues requiring physical
access to the machine.

## Best practices for users

- Store API keys in `.env` (mode 600), never in Task YAML or commit history.
- Review the `write_scope` and `shell_policy` of any Task YAML you import from
  an untrusted source before running it.
- Use `shell_policy: gated` (the default) so every shell command requires your
  explicit approval.
- Keep neurosurfer updated: `pip install --upgrade neurosurfer`.
