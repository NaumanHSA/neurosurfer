"""Context-compaction summary prompt and formatting.

The model is asked to produce an <analysis> drafting scratchpad (stripped before
the summary re-enters context) followed by a <summary> section with 9 numbered
sections covering the full conversation state.  formatCompactSummary() strips the
analysis block and unwraps the summary XML tags.
"""

from __future__ import annotations

import re

# ──────────────────────────────────────────────────────────────────────────────
# Ported verbatim from prompt.ts (NO_TOOLS_PREAMBLE)
# ──────────────────────────────────────────────────────────────────────────────
_NO_TOOLS_PREAMBLE = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use any tools whatsoever.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

# Ported from prompt.ts (NO_TOOLS_TRAILER)
_NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)

# 9-section prompt; ported from BASE_COMPACT_PROMPT in prompt.ts.
_BASE_COMPACT_PROMPT = """\
Your task is to create a detailed summary of the conversation so far, paying close attention \
to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural \
decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your \
thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section \
thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like file names, full code snippets, function signatures, file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback, especially if the user told you to do \
something differently.
2. Double-check for technical accuracy and completeness, addressing each required element \
thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks \
discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or \
created. Pay special attention to the most recent messages and include full code snippets where \
applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special \
attention to specific user feedback that you received.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for \
understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this \
summary request, paying special attention to the most recent messages from both user and \
assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent \
work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most \
recent explicit requests, and the task you were working on immediately before this summary \
request. Include direct quotes from the most recent conversation showing exactly what task you \
were working on and where you left off.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Important Code Snippet]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and \
ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, \
remember to follow these instructions when creating the above summary.
"""


def get_compact_prompt(
    custom_instructions: str | None = None,
    task_must_preserve: list[str] | None = None,
) -> str:
    """Build the full compaction user-message sent to the summarization call.

    ``task_must_preserve`` items are appended so the model knows what durable
    state to prioritize; ``custom_instructions`` extend the summary prompt with
    custom-instructions mechanism.
    """
    prompt = _NO_TOOLS_PREAMBLE + _BASE_COMPACT_PROMPT

    if task_must_preserve:
        items = "\n".join(f"- {item}" for item in task_must_preserve)
        prompt += (
            "\n\n## Items that MUST be preserved verbatim in the summary:\n" + items
        )

    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"

    prompt += _NO_TOOLS_TRAILER
    return prompt


def format_compact_summary(raw: str) -> str:
    """Strip the <analysis> scratchpad and unwrap <summary> tags.

    Ported from formatCompactSummary() in prompt.ts.  The <analysis> block is
    only a drafting scratchpad — it improves summary quality but has no
    informational value once the summary is written.
    """
    result = raw
    result = re.sub(r"<analysis>[\s\S]*?</analysis>", "", result)
    m = re.search(r"<summary>([\s\S]*?)</summary>", result)
    if m:
        content = (m.group(1) or "").strip()
        result = re.sub(r"<summary>[\s\S]*?</summary>", f"Summary:\n{content}", result)
    result = re.sub(r"\n\n+", "\n\n", result)
    return result.strip()


def get_compact_user_summary_message(raw_summary: str) -> str:
    """Wrap the formatted summary in the continuation-session preamble.

    Ported from getCompactUserSummaryMessage() in prompt.ts (suppress-follow-up
    variant — we always suppress, matching the auto-compact path).
    """
    formatted = format_compact_summary(raw_summary)
    return (
        "This session is being continued from a previous conversation that ran out of context. "
        "The summary below covers the earlier portion of the conversation.\n\n"
        f"{formatted}\n\n"
        "Continue the conversation from where it left off without asking the user any further "
        "questions. Resume directly — do not acknowledge the summary, do not recap what was "
        "happening, do not preface with \"I'll continue\" or similar. "
        "Pick up the last task as if the break never happened."
    )
