"""clarify — interactive Q&A function node.

Receives the ``discover`` node's structured output (a :class:`DiscoveryOutput` or
an equivalent dict / Pydantic model), presents each question to the user with its
3 choices, and returns a ``dict[str, str]`` mapping question-id → chosen answer.

Called by the GraphExecutor as a *function* node; kwargs are::

    clarify(user_intent=..., discover=<DiscoveryOutput | dict>)
"""

from __future__ import annotations

from typing import Any


def run(*, discover: Any, answers: dict[str, str] | None = None, **_: Any) -> dict[str, str]:  # noqa: D401
    """Present clarifying questions interactively and return answers dict.

    If *answers* is already provided (e.g. collected by a conversational
    front-end before the graph started), the interactive Q&A is skipped
    entirely and those answers are returned as-is.
    """
    if answers:
        return answers

    # Support both Pydantic model and plain dict (e.g. from JSON-mode output).
    if hasattr(discover, "questions"):
        questions = discover.questions
        summary = getattr(discover, "summary", "")
    elif isinstance(discover, dict):
        questions = discover.get("questions", [])
        summary = discover.get("summary", "")
    else:
        return {}

    if summary:
        print(f"\n[Architect] Problem summary:\n{summary}\n")

    answers: dict[str, str] = {}

    for q in questions:
        # Normalise: support both Pydantic model and dict
        if hasattr(q, "id"):
            qid = q.id
            question_text = q.question
            choices: list[str] = list(q.choices)
        else:
            qid = q.get("id", f"q{len(answers) + 1}")
            question_text = q.get("question", "")
            choices = list(q.get("choices", []))

        print(f"\n[Architect] {question_text}")
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")

        while True:
            raw = input("  Your choice (1-3): ").strip()
            if raw in {"1", "2", "3"} and int(raw) <= len(choices):
                answers[qid] = choices[int(raw) - 1]
                break
            # Allow typing the choice text directly too
            match = next(
                (c for c in choices if c.lower().startswith(raw.lower())), None
            )
            if match:
                answers[qid] = match
                break
            print(f"  Please enter a number 1–{len(choices)} or the start of a choice.")

    return answers
