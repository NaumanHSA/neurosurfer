from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class OpenAIBase(BaseModel):
    model_config = ConfigDict(extra="allow")


class ModelPermission(OpenAIBase):
    id: str
    object: Literal["model_permission"] = "model_permission"
    created: int
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(OpenAIBase):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "neurosurfer"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBase):
    object: Literal["list"] = "list"
    data: List[ModelCard]


class ChatMessage(OpenAIBase):
    role: str
    content: Any = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class ChatCompletionRequest(OpenAIBase):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    user: Optional[str] = None
    response_format: Optional[dict] = None
    stream_options: Optional[dict] = None
    metadata: Optional[dict] = None


class ChatCompletionChoice(OpenAIBase):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(OpenAIBase):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class ChatCompletionChunkDelta(OpenAIBase):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class ChatCompletionChunkChoice(OpenAIBase):
    index: int = 0
    delta: ChatCompletionChunkDelta = Field(default_factory=ChatCompletionChunkDelta)
    finish_reason: Optional[str] = None


class ChatCompletionChunk(OpenAIBase):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
