from typing import Any, Dict, Generator, Optional, Union
from pydantic import BaseModel as PydModel
from dataclasses import dataclass
from typing import List
from neurosurfer.tracing import TraceResult

@dataclass
class ToolCallResponse:
    selected_tool: str
    inputs: Dict[str, Any]
    returns: Union[str, Generator[str, None, None]]
    final: bool
    extras: Dict[str, Any]

@dataclass
class StructuredResponse:
    output_schema: PydModel
    model_response: str
    json_obj: Optional[str] = None
    parsed_output: Optional[PydModel] = None

@dataclass
class AgentResponse:
    response: Union[str, Generator[str, None, None], StructuredResponse, ToolCallResponse]
    traces: Optional[TraceResult] = None
