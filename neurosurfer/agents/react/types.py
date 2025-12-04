from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Generator, List
from neurosurfer.tracing import TraceResult
from neurosurfer.tools.base_tool import ToolResponse

@dataclass
class ToolCall:
    tool: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    final_answer: bool = False
    memory_keys: Optional[List[str]] = None
    # optional explanation for debug / self-repair prompts
    rationale: Optional[str] = None
    output: Optional[str] = None

@dataclass
class ReactAgentResponse:
    response: Union[str, Generator[str, None, None]]
    traces: Optional[TraceResult] = None
    tool_calls: Optional[List[ToolCall]] = None
