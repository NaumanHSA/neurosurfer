from typing import Union, Generator

from .model_registry import ModelCard, ModelList
from .model_response import ChatCompletionResponse, ChatCompletionChunk, Choice, ChoiceMessage, DeltaContent, StreamChoice, Usage
from .auth import User, LoginRequest, LoginResponse
from .completions import ChatCompletionRequest, ToolDef, ToolDefFunction, FileContent, ChatHandlerMessages, ChatHandlerModel
from .chats import Chat, ChatMessageIn, ChatMessageOut, ChatFileOut

AppResponseModel = Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]