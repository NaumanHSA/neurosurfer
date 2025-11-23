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
    thread_id: Optional[int] = None
    message_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    has_files: Optional[bool] = False

class ChatHandlerMessages(BaseModel):
    user_query: str
    user_msgs: List[str]
    assistant_msgs: List[str]
    system_msgs: List[str]
    converstaion: List[Dict[str, str]]

class ChatHandlerModel(BaseModel):
    model: str
    user_id: int
    thread_id: Optional[int] = None
    message_id: Optional[int] = None
    has_files_message: bool = False 
    messages: ChatHandlerMessages
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    system_prompt: Optional[str] = None
    tools: Optional[List[ToolDef]] = None
    tool_choice: Optional[str | Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None