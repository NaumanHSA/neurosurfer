from typing import List

class History:
    """
    A simple list of strings appended in the loop.
    """
    def __init__(self) -> None:
        self._h: List[str] = []

    def append(self, line: str) -> None:
        self._h.append(line)

    def as_text(self) -> str:
        return "\n".join(self._h)

    def __bool__(self):  # not used, but avoids accidental truthiness bugs
        return bool(self._h)

    def __len__(self):
        return len(self._h)

    def to_prompt(self) -> str:
        if not self._h:
            return ""
        out = "# Chain of Thoughts:\n"
        for h in self._h:
            out += f"{h}\n"
        return out
