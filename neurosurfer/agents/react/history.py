from typing import List

class History:
    """
    A simple list of strings appended in the loop.
    Each entry is typically a block like:
      - Thought: ...
      - Action: {...}
      - results: ...
    """
    def __init__(self) -> None:
        self._h: List[str] = []

    def append(self, line: str) -> None:
        self._h.append(line)

    def as_text(self) -> str:
        return "\n".join(self._h)

    def __bool__(self):
        return bool(self._h)

    def __len__(self):
        return len(self._h)

    def to_prompt(self) -> str:
        """
        Render the previous reasoning steps as a readable history.

        We emphasize that these are *past* steps so the model is
        less tempted to repeat them verbatim.
        """
        if not self._h:
            return ""

        out = "\n# Chain of Thoughts (previous steps, already completed):\n"
        for i, h in enumerate(self._h, start=1):
            out += f"Step {i}:\n{h}\n"
        out += "# End of previous steps.\n\n"
        return out
