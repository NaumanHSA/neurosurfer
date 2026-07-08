# Contributing to Neurosurfer

Thank you not only for using **Neurosurfer**, but also for being interested in helping it grow! We welcome contributions in all forms — code, ideas, docs, testing, or simply by spreading the word. Your participation makes the ecosystem better for everyone. 💙

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

## Pull Requests (PRs)

PRs are very welcome — small and focused is best. Before you open one:

- Make sure your change builds locally and tests pass.
- Include tests for new behavior when possible.
- Update docs if user‑facing behavior changes.
- Keep the description short but clear about *what* and *why*.

We review for correctness, clarity, and scope. Friendly feedback is normal; we appreciate your iteration!

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

- Issues → bugs and feature requests  
- Discussions → questions, design ideas, showcases  
- Security → email naumanhsa965@gmail.com

---

## License

By contributing, you agree that your contributions are provided under the project license (**Apache‑2.0**).

Thank you for helping improve Neurosurfer — we’re excited to see what you build! 🙌
