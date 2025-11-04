from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ToolCall:
    tool: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    final_answer: bool = False
    # optional explanation for debug / self-repair prompts
    rationale: Optional[str] = None
