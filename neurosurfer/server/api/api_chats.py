# server/api/chats_api.py
"""
Chat Management API Module
===========================

This module provides FastAPI routes for managing chat threads and messages.

Endpoints:
    - GET /chats: List all chat threads for current user
    - POST /chats: Create a new chat thread
    - GET /chats/{chat_id}: Get specific chat thread
    - GET /chats/{chat_id}/messages: List messages in a thread
    - POST /chats/{chat_id}/messages: Add message to a thread
    - PUT /chats/{chat_id}: Update chat thread (title)
    - DELETE /chats/{chat_id}: Delete chat thread and all messages

Features:
    - User-scoped chat threads
    - Automatic title generation from first message
    - Message ordering by timestamp
    - Thread metadata (message count, timestamps)
    - Cascade deletion of messages
    - Efficient queries with joins and aggregations

All endpoints require authentication and only return data
belonging to the authenticated user.

Example:
    >>> # Create thread
    >>> POST /chats
    >>> {"title": "My Chat"}
    >>> # Returns: {"id": "1", "title": "My Chat", ...}
    >>> 
    >>> # Add message
    >>> POST /chats/1/messages
    >>> {"role": "user", "content": "Hello"}
    >>> 
    >>> # List messages
    >>> GET /chats/1/messages
    >>> # Returns: [{"id": 1, "role": "user", "content": "Hello", ...}]
"""
import os
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from ..security import get_db, get_current_user
from ..db.models import User, ChatThread, Message, NMFile
from ..schemas import Chat, ChatMessage
from ..schemas import Chat, ChatMessage, ChatMessageOut 

router = APIRouter(prefix="/chats", tags=["chats"])

def thread_to_chat(th: ChatThread) -> Chat:
    """
    Convert ChatThread model to Chat schema.
    
    Args:
        th (ChatThread): Database chat thread model
    
    Returns:
        Chat: API chat schema
    """
    return Chat(
        id=str(th.id),
        title=th.title or "New Chat",
        createdAt=int(th.created_at.timestamp()),
        updatedAt=int(th.updated_at.timestamp()),
        messagesCount=len(th.messages) if hasattr(th, "messages") else 0,
    )

def _row_to_chat(th: ChatThread, msg_count: int, last_ts) -> Chat:
    """
    Convert query result row to Chat schema.
    
    Helper for converting aggregated query results (with message counts)
    to API schema format.
    
    Args:
        th (ChatThread): Database chat thread model
        msg_count (int): Number of messages in thread
        last_ts: Timestamp of last message
    
    Returns:
        Chat: API chat schema with metadata
    """
    return Chat(
        id=str(th.id),
        title=th.title or "New Chat",
        createdAt=int(th.created_at.timestamp()),
        updatedAt=int((last_ts or th.created_at).timestamp()),
        messagesCount=int(msg_count or 0),
    )

# This endpoint is used to get a list of chat threads
@router.get("", response_model=List[Chat])
def list_threads(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    List all chat threads for the current user.
    
    Returns threads ordered by most recent activity (last message timestamp).
    Includes message count and timestamps for each thread.
    
    Args:
        db (Session): Database session
        user (User): Current authenticated user
    
    Returns:
        List[Chat]: List of chat threads with metadata
    
    Example:
        >>> GET /chats
        >>> # Returns: [{"id": "1", "title": "Chat 1", "messagesCount": 5, ...}]
    """
    q = (
        db.query(
            ChatThread,
            func.count(Message.id).label("msg_count"),
            func.max(Message.created_at).label("last_ts"),
        )
        .outerjoin(Message, Message.thread_id == ChatThread.id)
        .filter(ChatThread.user_id == user.id)
        .group_by(ChatThread.id)
        .order_by(func.coalesce(func.max(Message.created_at), ChatThread.created_at).desc())
    )
    rows = q.all()
    return [_row_to_chat(th, msg_count, last_ts) for (th, msg_count, last_ts) in rows]

# This endpoint is used to create a new chat thread
@router.post("", response_model=Chat, status_code=status.HTTP_201_CREATED)
def create_thread(data: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    Create a new chat thread.
    
    Args:
        data (dict): Thread data (optional title)
        db (Session): Database session
        user (User): Current authenticated user
    
    Returns:
        Chat: Created chat thread
    
    Example:
        >>> POST /chats
        >>> {"title": "My New Chat"}
        >>> # Returns: {"id": "1", "title": "My New Chat", ...}
    """
    title = (data or {}).get("title") or "New Chat"
    th = ChatThread(user_id=user.id, title=title)
    db.add(th)
    db.commit()
    db.refresh(th)
    return thread_to_chat(th)

# This endpoint is used to get a chat thread
@router.get("/{chat_id}", response_model=Chat)
def get_thread(chat_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    Get a specific chat thread.
    
    Args:
        chat_id (int): Chat thread ID
        db (Session): Database session
        user (User): Current authenticated user
    
    Returns:
        Chat: Chat thread details
    
    Raises:
        HTTPException: 404 if chat not found or doesn't belong to user
    
    Example:
        >>> GET /chats/1
        >>> # Returns: {"id": "1", "title": "My Chat", ...}
    """
    th = db.query(ChatThread).filter(ChatThread.id == chat_id, ChatThread.user_id == user.id).first()
    if not th: raise HTTPException(status_code=404, detail="Chat not found")
    return thread_to_chat(th)

# This endpoint is used to get a list of messages in a chat thread
@router.get("/{chat_id}/messages", response_model=List[ChatMessageOut])
def list_messages(chat_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    List all messages in a chat thread.
    
    Returns messages ordered chronologically (oldest first).
    
    Args:
        chat_id (int): Chat thread ID
        db (Session): Database session
        user (User): Current authenticated user
    
    Returns:
        List[ChatMessageOut]: List of messages in the thread
    
    Raises:
        HTTPException: 404 if chat not found or doesn't belong to user
    
    Example:
        >>> GET /chats/1/messages
        >>> # Returns: [{"id": 1, "role": "user", "content": "Hi", ...}]
    """
    th = db.query(ChatThread).filter(ChatThread.id == chat_id, ChatThread.user_id == user.id).first()
    if not th: raise HTTPException(status_code=404, detail="Chat not found")

    rows = (
        db.query(Message)
        .filter(Message.thread_id == th.id)
        .order_by(Message.created_at.asc())  # ✅ oldest → newest
        .all()
    )
    return [
        ChatMessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            createdAt=int(m.created_at.timestamp()),
        )
        for m in rows
    ]

# This endpoint is used to append a message to a chat thread
@router.post("/{chat_id}/messages", response_model=ChatMessage, status_code=status.HTTP_201_CREATED)
def append_message(chat_id: int, body: ChatMessage, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    Add a message to a chat thread.
    Automatically generates thread title from first user message if not set.
    Args:
        chat_id (int): Chat thread ID
        body (ChatMessage): Message data (role and content)
        db (Session): Database session
        user (User): Current authenticated user
    
    Returns:
        ChatMessage: Created message
    
    Raises:
        HTTPException: 404 if chat not found or doesn't belong to user
    
    Example:
        >>> POST /chats/1/messages
        >>> {"role": "user", "content": "Hello, how are you?"}
        >>> # Returns: {"role": "user", "content": "Hello, how are you?"}
    """
    th = db.query(ChatThread).filter(ChatThread.id == chat_id, ChatThread.user_id == user.id).first()
    if not th: raise HTTPException(status_code=404, detail="Chat not found")

    # Auto-title on first user message
    if (th.title or "New Chat") == "New Chat" and body.role == "user":
        first_line = (body.content or "").strip().splitlines()[0][:60]
        if first_line:
            th.title = first_line

    msg = Message(thread_id=th.id, role=body.role, content=body.content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return ChatMessage(role=msg.role, content=msg.content)

# This endpoint is used to delete a chat thread
@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(chat_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    th = db.query(ChatThread).filter(ChatThread.id == chat_id, ChatThread.user_id == user.id).first()
    if not th:
        raise HTTPException(status_code=404, detail="Chat not found")

    # first delete all files stored for this thread
    for file in th.files:
        if os.path.exists(file.stored_path):
            os.remove(file.stored_path)

    db.delete(th)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# endpoint to update chat thread title
@router.put("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def update_thread(chat_id: int, data: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    th = db.query(ChatThread).filter(ChatThread.id == chat_id, ChatThread.user_id == user.id).first()
    if not th:
        raise HTTPException(status_code=404, detail="Chat not found")
    title = (data or {}).get("title") or "New Chat"
    th.title = title
    th.updated_at = func.now()
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


