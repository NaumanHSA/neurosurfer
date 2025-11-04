from typing import Dict, Any

class EphemeralMemory:
    """
    Short-lived scratch memory for passing small items between steps.
    Cleared after each tool execution (by agent).
    """
    def __init__(self):
        self._mem: Dict[str, Any] = {}

    def set(self, key: str, val: Any) -> None:
        self._mem[key] = val

    def items(self) -> Dict[str, Any]:
        return dict(self._mem)

    def clear(self) -> None:
        self._mem = {}
