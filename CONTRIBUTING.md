# Contributing to Neurosurfer

Thank you not only for using **Neurosurfer**, but also for being interested in helping it grow! We welcome contributions in all forms — code, ideas, docs, testing, or simply by spreading the word. Your participation makes the ecosystem better for everyone. 💙

---

## 🚩 Important: Pull Requests Target `dev`

Please open PRs **only against the `dev` branch**.  
PRs targeting `main` are not accepted and may be auto-closed by CI. We promote stable changes from `dev` → `main` during release cycles.

**Quick example:**

```bash
# Good
base: dev   ← your-username:feature/add-rag-agent

# Avoid
base: main  ← will be rejected
```

---

## How You Can Help

There isn’t just one way to contribute. Some people dive into code; others lift the community by sharing knowledge. All of it matters.

- **Support the community** — answer questions, review small PRs, and share quick tips in discussions.
- **Fix bugs** — reproduce the issue, include a minimal snippet, and submit a clear, targeted fix.
- **Share ideas** — describe the problem, the desired outcome, and any alternatives you considered.
- **Build features** — open an issue to align scope, then ship a focused, well-tested PR.
- **Improve docs** — add concise examples, clarify tricky spots, and keep instructions accurate and current.
- **Spread the word** — write posts, give demos, or simply ⭐ the repo to help others find it.

---

## Submitting Issues & Ideas

We love actionable, well‑scoped reports and proposals. A great issue saves everyone time:

- **Search first.** Please check if your topic already exists in Issues/Discussions.
- **Add context.** Where are you running (local, Docker, Colab, Kaggle)? Which OS/Python/Neurosurfer version? Which model/backends?
- **Be specific.** For bugs, include a minimal snippet (or steps) to reproduce. Screenshots, stack traces, or logs are extremely helpful.
- **Stay focused.** One issue per topic keeps the conversation crisp and the fix targeted.

If it’s a **feature request**, explain the problem you’re solving, the desired outcome, and any alternatives you considered. A tiny pseudo‑API example is perfect.

---

## Branch Model

| Branch | Purpose | Who pushes | Notes |
|-------|---------|------------|-------|
| `main` | Stable, production, releases (docs & PyPI) | Maintainers only | Protected; PRs from `dev` only |
| `dev`  | Active development | Contributors via PR | Default target for all PRs |
| `feature/*` | Work-in-progress features | Contributors | Open PRs into `dev` |
| `hotfix/*` | Urgent fixes for `main` | Maintainers | Promoted to `main` with a patch release |

> Maintainers may keep experimental folders (e.g., `labs/`) on `dev` only; workflows prevent them from landing in `main`.

---

## Pull Requests (PRs)

PRs are very welcome — small and focused is best. Before you open one:

1. **Fork** the repo and create a topic branch from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/short-name
   ```
2. **Implement** your change; keep commits tidy (use meaningful messages).
3. **Run checks** locally (tests, lint, type checks).
4. **Target `dev`** when opening the PR. Describe *what* and *why* briefly.
5. **Update docs** if user‑facing behavior changes.
6. **Add tests** for new behavior whenever possible.

**We review for** correctness, clarity, scope, and tests. Friendly iteration is normal — thank you for working with us!

---

## Development Quickstart

If you want to run Neurosurfer locally for development or to try changes:

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/<your-username>/neurosurfer.git
cd neurosurfer

# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install core (pick extras as needed)
pip install -U pip
pip install -e .
# Optional stacks:
# pip install -e '.[torch]'      # local model stack

# Run tests
pytest -q

# Lint & format
black neurosurfer tests && ruff check --fix .
```

If you’re working on the docs:

```bash
mkdocs serve   # preview locally at http://127.0.0.1:8000
```

---

## Code Style & Testing

- **Style:** Black for formatting, Ruff for linting, MyPy for typing on changed files.
- **Tests:** Keep them minimal, readable, and fast. If you fix a bug, add a test that would have caught it.
- **Scope:** Prefer smaller PRs with clear intent over giant changes that are harder to review.

---

## Communication & Conduct

We aim for a welcoming, inclusive space. Be respectful, kind, and constructive. If you see something that could be better, propose a change — we’re all building this together.

- **Issues** → bugs and feature requests  
- **Discussions** → questions, design ideas, showcases  
- **Security** → email naumanhsa965@gmail.com

---

## License

By contributing, you agree that your contributions are provided under the project license (**Apache‑2.0**).

Thank you for helping improve Neurosurfer — we’re excited to see what you build! 🙌