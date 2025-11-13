from typing import Any, Dict, Generator, Optional, Union
from pydantic import BaseModel as PydModel
from dataclasses import dataclass

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