from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Generator
from neurosurfer.tracing import TraceResult

@dataclass
class ToolCall:
    tool: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    final_answer: bool = False
    # optional explanation for debug / self-repair prompts
    rationale: Optional[str] = None

@dataclass
class ReactAgentResponse:
    response: Union[str, Generator[str, None, None]]
    traces: Optional[TraceResult] = None
