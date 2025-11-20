from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ============ Chat ============
class Chat(BaseModel):
    id: str
    title: str
    createdAt: int
    updatedAt: int
    messagesCount: int

class UploadedFileIn(BaseModel):
    name: str
    mime: Optional[str] = None
    size: Optional[int] = None
    base64: str

class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    files: List[UploadedFileIn] = Field(default_factory=list)

class ChatFileOut(BaseModel):
    id: str
    filename: str
    mime: Optional[str] = None
    size: Optional[int] = None
    downloadUrl: str

class ChatMessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant", "system"]
    content: str
    createdAt: int
    files: List[ChatFileOut] = Field(default_factory=list)
