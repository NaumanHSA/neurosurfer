from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

# ============ Completion Request ============
class ToolDefFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ToolDef(BaseModel):
    type: Literal["function"]
    function: ToolDefFunction

class FileContent(BaseModel):
    name: str
    content: str                 # base64-encoded string
    type: Optional[str] = None   # e.g. "application/pdf"
    
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    tools: Optional[List[ToolDef]] = None
    tool_choice: Optional[str | Dict[str, Any]] = None
    files: Optional[List[FileContent]] = None
    thread_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
