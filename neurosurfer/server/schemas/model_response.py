from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import time

# ============ Non-Streaming Response Models ============

class ChoiceMessage(BaseModel):
    """Message in a non-streaming completion response"""
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """Choice in a non-streaming completion response"""
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Complete non-streaming chat completion response"""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)


# ============ Streaming Response Models ============

class DeltaContent(BaseModel):
    """Delta content for streaming chunks"""
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """Choice in a streaming chunk"""
    index: int = 0
    delta: DeltaContent
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response"""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
