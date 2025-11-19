"""
Database Models Module
======================

This module defines SQLAlchemy ORM models for the Neurosurfer server database.

Models:
    - User: User accounts with authentication
    - ChatThread: Conversation threads belonging to users
    - Message: Individual messages within chat threads
    - NMFile: File attachments associated with threads

Relationships:
    - User -> ChatThread (one-to-many)
    - ChatThread -> Message (one-to-many)
    - ChatThread -> NMFile (one-to-many)
    - User -> NMFile (one-to-many)

All models include automatic timestamps and proper indexing for performance.
Cascade deletes ensure data integrity when users or threads are removed.

Example:
    >>> from neurosurfer.server.db.models import User, ChatThread, Message
    >>> from neurosurfer.server.db.db import SessionLocal
    >>> 
    >>> db = SessionLocal()
    >>> user = User(email="user@example.com", hashed_password="...")
    >>> db.add(user)
    >>> db.commit()
    >>> 
    >>> thread = ChatThread(user_id=user.id, title="My Chat")
    >>> db.add(thread)
    >>> db.commit()
"""
from datetime import datetime
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy import String, Integer, Text, DateTime, ForeignKey, func, Index, Column
from .db import Base

class User(Base):
    """
    User account model.
    
    Represents a registered user with authentication credentials.
    Users can have multiple chat threads and files.
    
    Attributes:
        id (int): Primary key
        email (str): Unique email address (indexed)
        full_name (str | None): Optional full name
        hashed_password (str): Bcrypt hashed password
        created_at (datetime): Account creation timestamp
        threads (list[ChatThread]): User's chat threads (cascade delete)
    
    Indexes:
        - email (unique)
        - id (primary key)
    
    Example:
        >>> user = User(
        ...     email="user@example.com",
        ...     full_name="John Doe",
        ...     hashed_password=hash_password("password")
        ... )
    """
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    threads: Mapped[list["ChatThread"]] = relationship("ChatThread", back_populates="user", cascade="all,delete")

class ChatThread(Base):
    """
    Chat thread (conversation) model.
    
    Represents a conversation thread containing multiple messages.
    Each thread belongs to a user and can have associated files.
    
    Attributes:
        id (int): Primary key
        user_id (int): Foreign key to User (indexed)
        title (str | None): Thread title (auto-generated from first message)
        created_at (datetime): Thread creation timestamp
        updated_at (datetime): Last update timestamp (auto-updated)
        user (User): Thread owner
        messages (list[Message]): Messages in this thread (cascade delete, ordered by created_at)
        files (list[NMFile]): Files attached to this thread (cascade delete)
    
    Indexes:
        - id (primary key)
        - user_id
        - (user_id, created_at) composite index
    
    Example:
        >>> thread = ChatThread(
        ...     user_id=1,
        ...     title="Discussion about AI"
        ... )
    """
    __tablename__ = "chat_threads"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship("User", back_populates="threads")
    messages: Mapped[list["Message"]] = relationship("Message", back_populates="thread", cascade="all,delete", order_by="Message.created_at")
    files: Mapped[list["NMFile"]] = relationship("NMFile", back_populates="thread", cascade="all,delete", order_by="NMFile.created_at")

    __table_args__ = (
        Index("ix_thread_user_created", "user_id", "created_at"),
    )

class Message(Base):
    """
    Chat message model.
    
    Represents a single message within a chat thread.
    Messages have a role (user, assistant, system, tool) and content.
    
    Attributes:
        id (int): Primary key
        thread_id (int): Foreign key to ChatThread (indexed)
        role (str): Message role (user | assistant | system | tool)
        content (str): Message text content
        model_name (str | None): Model used to generate this message (if assistant)
        created_at (datetime): Message creation timestamp
        thread (ChatThread): Parent thread
    
    Indexes:
        - id (primary key)
        - thread_id
        - (thread_id, created_at) composite index
    
    Example:
        >>> message = Message(
        ...     thread_id=1,
        ...     role="user",
        ...     content="Hello, how are you?"
        ... )
    """
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    thread_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_threads.id"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # user | assistant | system | tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    thread: Mapped["ChatThread"] = relationship("ChatThread", back_populates="messages")

    __table_args__ = (
        Index("ix_message_thread_created", "thread_id", "created_at"),
    )


# ---------- NEW ----------
class NMFile(Base):
    """
    File attachment model.
    
    Represents a file uploaded by a user and associated with a chat thread.
    Files are ingested into vector stores for RAG functionality.
    
    Attributes:
        id (str): Primary key (file ID)
        user_id (int): Foreign key to User (indexed)
        thread_id (int): Foreign key to ChatThread (indexed)
        filename (str): Original filename
        stored_path (str): Path where file is stored (may be removed after ingest)
        mime (str | None): MIME type (e.g., "application/pdf")
        size (int | None): File size in bytes
        collection (str): Vector store collection name for this file
        created_at (datetime): Upload timestamp
        thread (ChatThread): Parent thread
    
    Indexes:
        - id (primary key)
        - user_id
        - thread_id
    
    Example:
        >>> file = NMFile(
        ...     id="file_abc123",
        ...     user_id=1,
        ...     thread_id=1,
        ...     filename="document.pdf",
        ...     stored_path="/uploads/document.pdf",
        ...     mime="application/pdf",
        ...     size=1024000,
        ...     collection="nm_u1_t1"
        ... )
    """
    __tablename__ = "nm_files"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    thread_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_threads.id"), index=True, nullable=False)

    filename: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str | None] = mapped_column(String, nullable=True)
    stored_path: Mapped[str] = mapped_column(String, nullable=False)   # may be removed after ingest
    mime: Mapped[str | None] = mapped_column(String, nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    collection: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    thread: Mapped["ChatThread"] = relationship("ChatThread", back_populates="files")