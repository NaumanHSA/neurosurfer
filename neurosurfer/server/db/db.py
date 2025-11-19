"""
Database Connection Module
===========================

This module provides SQLAlchemy database connection and session management
for the Neurosurfer server.

Components:
    - engine: SQLAlchemy engine for database connections
    - SessionLocal: Session factory for creating database sessions
    - Base: Declarative base for ORM models
    - init_db(): Initialize database tables

The module automatically handles SQLite directory creation and configures
appropriate connection arguments for different database backends.

Example:
    >>> from neurosurfer.server.db.db import SessionLocal, init_db
    >>> 
    >>> # Initialize database tables
    >>> init_db()
    >>> 
    >>> # Create a session
    >>> db = SessionLocal()
    >>> try:
    ...     # Perform database operations
    ...     user = db.query(User).first()
    ... finally:
    ...     db.close()
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import logging
from ..config import DATABASE_URL

logger = logging.getLogger("neurosurfer")

# Ensure SQLite directory exists
if DATABASE_URL.startswith("sqlite:///"):
    path = DATABASE_URL.replace("sqlite:///", "")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """
    Initialize database tables.
    
    Creates all tables defined in models if they don't exist.
    This function should be called once during application startup.
    
    Example:
        >>> from neurosurfer.server.db.db import init_db
        >>> init_db()
        Database initialized successfully...
    """
    from . import models  # ensure models are imported
    # from .models import Base as _Base
    # _Base.metadata.create_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully in %s", DATABASE_URL)

