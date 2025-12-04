# neurosurfer/server/api/chat_completions.py
"""
Chat Completions API Module
============================

This module provides OpenAI-compatible chat completion endpoints for the Neurosurfer server.
It handles both streaming and non-streaming chat requests with support for custom chat handlers.

Features:
    - OpenAI-compatible API format
    - Streaming and non-streaming responses
    - Server-Sent Events (SSE) for streaming
    - Operation lifecycle management
    - Model validation
    - User authentication (optional)
    - Request context tracking

Endpoints:
    - POST /chat/completions: Create chat completion (streaming or non-streaming)

The module supports flexible chat handler registration, allowing custom implementations
while maintaining API compatibility. Handlers can be synchronous or asynchronous,
and can return various response formats (Pydantic models, dicts, or plain strings).

Example:
    >>> # Register a chat handler
    >>> @app.chat()
    >>> def my_handler(request: ChatCompletionRequest, context: RequestContext):
    ...     return "Hello, " + request.messages[-1]["content"]
    >>> 
    >>> # Client request
    >>> POST /chat/completions
    >>> {
    ...     "model": "gpt-4",
    ...     "messages": [{"role": "user", "content": "Hi"}],
    ...     "stream": false
    ... }
"""
from __future__ import annotations
import json, time, inspect, asyncio
from typing import Any, AsyncGenerator, Union
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import ORJSONResponse, StreamingResponse

from ..schemas import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionChunk,
    Choice, 
    ChoiceMessage, 
    Usage,
    ChatHandlerModel,
    ChatHandlerMessages
)
from ..runtime import op_manager, RequestContext
from ..models_registry import ModelRegistry
from ..security import maybe_current_user, resolve_actor_id
from ..services.follow_up_questions import FollowUpQuestions
from ..services.thread_title_generator import ThreadTitleGenerator
from neurosurfer.config import config


def _sse_chunk(obj: Union[ChatCompletionChunk, dict, str]) -> bytes:
    """
    Convert response object to Server-Sent Events (SSE) format.
    
    Handles multiple input types (Pydantic models, dicts, strings) and
    converts them to SSE-formatted bytes for streaming responses.
    
    Args:
        obj (Union[ChatCompletionChunk, dict, str]): Response object to convert
    
    Returns:
        bytes: SSE-formatted data (e.g., "data: {...}\\n\\n")
    
    Example:
        >>> chunk = ChatCompletionChunk(...)
        >>> sse_data = _sse_chunk(chunk)
        >>> print(sse_data.decode())
        data: {"id": "...", "choices": [...]}
    """
    if isinstance(obj, ChatCompletionChunk):
        # Pydantic model - use model_dump()
        data = obj.model_dump(exclude_none=True)
    elif isinstance(obj, dict):
        data = obj
    elif isinstance(obj, str):
        try:
            data = json.loads(obj)
        except:
            data = {"choices": [{"delta": {"content": obj}}]}
    else:
        data = {"choices": [{"delta": {"content": str(obj)}}]}
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

def make_chat_response(op_id: str, model: str, text: str) -> ChatCompletionResponse:
    """
    Create OpenAI-compatible chat completion response from plain text.
    
    Helper function to wrap simple text responses in the proper Pydantic model format.
    
    Args:
        op_id (str): Operation/request ID
        model (str): Model identifier
        text (str): Response text content
    
    Returns:
        ChatCompletionResponse: Formatted response with usage stats
    
    Example:
        >>> response = make_chat_response("op_123", "gpt-4", "Hello!")
        >>> print(response.choices[0].message.content)
        'Hello!'
    """
    return ChatCompletionResponse(
        id=op_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0, 
                message=ChoiceMessage(role="assistant", content=text)
            )
        ],
        usage=Usage()
    )


def chat_completion_router(_chat_handler, model_registry: ModelRegistry):
    """
    Create FastAPI router for chat completion endpoints.
    
    Factory function that creates a router with the /chat/completions endpoint,
    configured with the provided chat handler and model registry.
    
    Args:
        _chat_handler: Custom chat handler function (sync or async)
        model_registry (ModelRegistry): Registry of available models
    
    Returns:
        APIRouter: Configured FastAPI router
    
    Example:
        >>> def my_handler(request, context):
        ...     return "Response"
        >>> router = chat_completion_router(my_handler, registry)
        >>> app.include_router(router)
    """
    router = APIRouter(prefix="", tags=["Chat"])
    follow_up_questions_service = FollowUpQuestions()
    thread_title_generator = ThreadTitleGenerator()

    @router.post("/chat/completions")
    async def chat_completions(
        req: Request, 
        body: ChatCompletionRequest, 
        user=Depends(maybe_current_user)
    ):
        """
        Handle chat completion requests (streaming and non-streaming).
        
        OpenAI-compatible endpoint that processes chat requests through
        the registered chat handler. Supports both streaming (SSE) and
        non-streaming responses.
        
        Args:
            req (Request): FastAPI request object
            body (ChatCompletionRequest): Chat completion request data
            user: Current user (optional, via dependency)
        
        Returns:
            Union[ORJSONResponse, StreamingResponse]: Chat completion response
        
        Raises:
            HTTPException: 501 if no chat handler registered
            HTTPException: 404 if model not found
        
        Example:
            >>> POST /chat/completions
            >>> {
            ...     "model": "gpt-4",
            ...     "messages": [{"role": "user", "content": "Hello"}],
            ...     "stream": true
            ... }
        """
        if not _chat_handler:
            raise HTTPException(status_code=501, detail="No chat handler registered.")

        if not model_registry.exists(body.model):
            raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found.")

        # Add follow-up questions if requested from the UI - handle accordingly
        if body.metadata and body.metadata.get("follow_up_questions", False):
            # lazy init
            if not follow_up_questions_service.llm:
                follow_up_questions_service.set_llm(model_registry.get_first_available().llm)
            return follow_up_questions_service.generate(body.messages)

        # Add follow-up questions if requested from the UI - handle accordingly
        if body.metadata and body.metadata.get("generate_title", False):
            # lazy init
            thread_title_generator.set_llm(model_registry.get_first_available().llm)
            return thread_title_generator.generate(body.messages)

        # Create request context
        ctx: RequestContext = op_manager.create()
        ctx.headers = dict(req.headers)
        ctx.meta = getattr(ctx, "meta", {}) or {}
        ctx.user_id = user.id if user else None
        
        actor_id = resolve_actor_id(req, user)
        ctx.meta["actor_id"] = actor_id
        ctx.meta["thread_id"] = body.thread_id or None

        # validations
        temperature = body.temperature if (body.temperature and 2 > body.temperature > 0) else config.base_model.temperature
        max_tokens = body.max_tokens if (body.max_tokens and body.max_tokens > 512) else config.base_model.max_new_tokens
        system_msgs = [m["content"] for m in body.messages if m["role"] == "system"]
        user_msgs = [m["content"] for m in body.messages if m["role"] == "user"]
        conversation_messages = [msg for msg in body.messages if msg["role"] != "system"]
        system_prompt = system_msgs[0] if system_msgs else config.base_model.system_prompt    # First system message or default

        handler_args = ChatHandlerModel(
            user_id=user.id if user else None,
            thread_id=body.thread_id,
            message_id=body.message_id,
            has_files_message=body.has_files,
            model=body.model,
            messages=ChatHandlerMessages(
                user_query=user_msgs[-1] if user_msgs else "",    # Last user message
                user_msgs=user_msgs,
                assistant_msgs=[m["content"] for m in body.messages if m["role"] == "assistant"],
                system_msgs=system_msgs,
                converstaion=conversation_messages,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=body.stream,
            system_prompt=system_prompt,
        )

        body.temperature = temperature
        body.max_tokens = max_tokens

        response = _chat_handler(handler_args)

        # ============ Non-Streaming Mode ============
        if not body.stream:
            if inspect.iscoroutine(response):
                response = await response
            
            op_manager.done(ctx.op_id)
            
            # Handle different return types
            if isinstance(response, ChatCompletionResponse):
                # Pydantic model - use model_dump()
                return ORJSONResponse(response.model_dump())
            elif isinstance(response, dict):
                # Legacy dict response
                return ORJSONResponse(response)
            elif isinstance(response, str):
                # Plain string - wrap in Pydantic model
                response = make_chat_response(ctx.op_id, body.model, response)
                return ORJSONResponse(response.model_dump())
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Unsupported chat handler return type."
                )

        # ============ Streaming Mode ============
        async def gen(out) -> AsyncGenerator[bytes, None]:
            try:
                # Handle async generator
                if inspect.isasyncgen(out):
                    agen = out
                # Handle sync generator
                elif inspect.isgenerator(out):
                    async def to_async(gen):
                        for x in gen:
                            yield x
                    agen = to_async(out)
                # Handle coroutine
                elif inspect.iscoroutine(out):
                    out = await out
                    if inspect.isgenerator(out):
                        async def to_async2(gen):
                            for x in gen:
                                yield x
                        agen = to_async2(out)
                    else:
                        # Single response
                        yield _sse_chunk(out)
                        yield b"data: [DONE]\n\n"
                        return
                else:
                    # Direct result
                    yield _sse_chunk(out)
                    yield b"data: [DONE]\n\n"
                    return

                # Stream chunks
                async for piece in agen:
                    # piece can be ChatCompletionChunk (Pydantic), dict, or str
                    yield _sse_chunk(piece)
                    await asyncio.sleep(0)
                
                yield b"data: [DONE]\n\n"
            
            except Exception as e:
                # Log error and send error chunk
                import traceback
                traceback.print_exc()
                error_data = {
                    "error": {
                        "message": str(e),
                        "type": "internal_error"
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode("utf-8")
            finally:
                op_manager.done(ctx.op_id)

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
        return StreamingResponse(gen(response), media_type="text/event-stream", headers=headers)
    
    return router

