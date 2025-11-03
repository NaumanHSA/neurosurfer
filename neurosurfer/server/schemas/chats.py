from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# ============ Chat ============
class Chat(BaseModel):
    id: str
    title: str
    createdAt: int
    updatedAt: int
    messagesCount: int

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatMessageOut(ChatMessage):
    id: int
    createdAt: int

