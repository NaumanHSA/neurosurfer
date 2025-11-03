# server/security.py
"""
Server Security Module
======================

This module provides authentication and security utilities for the Neurosurfer server,
including password hashing, JWT token management, cookie handling, and user authentication.

Key Features:
    - Password hashing with bcrypt
    - JWT token creation and validation
    - Secure cookie management
    - Sliding session refresh
    - Database session management
    - Current user dependency injection

The module supports both header-based (Bearer token) and cookie-based authentication,
with automatic token refresh for active sessions.

Security Components:
    - Password hashing and verification
    - JWT token encoding/decoding
    - Cookie management (HttpOnly, Secure, SameSite)
    - User authentication with sliding refresh
    - Database session lifecycle

Example:
    >>> from neurosurfer.server.security import hash_password, verify_password
    >>> from neurosurfer.server.security import create_access_token
    >>> 
    >>> # Hash password
    >>> hashed = hash_password("my-password")
    >>> 
    >>> # Verify password
    >>> is_valid = verify_password("my-password", hashed)
    >>> print(is_valid)
    True
    >>> 
    >>> # Create JWT token
    >>> token = create_access_token({"sub": "user123"})
    >>> 
    >>> # Use in FastAPI endpoint
    >>> from fastapi import Depends
    >>> from neurosurfer.server.security import get_current_user
    >>> 
    >>> @app.get("/protected")
    >>> def protected_route(user = Depends(get_current_user)):
    ...     return {"user_id": user.id}
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import HTTPException, status, Response, Request, Depends
from sqlalchemy.orm import Session

from .config import (
    SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES,
    COOKIE_NAME, COOKIE_SECURE, COOKIE_SAMESITE
)
from .db.db import SessionLocal
from .db.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"

# ---------- DB session ----------
def get_db():
    """
    Database session dependency for FastAPI.
    
    Creates a new database session for each request and ensures
    it's properly closed after the request completes.
    
    Yields:
        Session: SQLAlchemy database session
    
    Example:
        >>> from fastapi import Depends
        >>> from neurosurfer.server.security import get_db
        >>> 
        >>> @app.get("/users")
        >>> def list_users(db: Session = Depends(get_db)):
        ...     return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Passwords ----------
def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password (str): Plain text password to hash
    
    Returns:
        str: Bcrypt hashed password
    
    Example:
        >>> hashed = hash_password("my-secure-password")
        >>> print(hashed[:7])
        '$2b$12$'
    """
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain (str): Plain text password to verify
        hashed (str): Bcrypt hashed password to compare against
    
    Returns:
        bool: True if password matches, False otherwise
    
    Example:
        >>> hashed = hash_password("password123")
        >>> verify_password("password123", hashed)
        True
        >>> verify_password("wrong-password", hashed)
        False
    """
    return pwd_context.verify(plain, hashed)

# ---------- Tokens ----------
def _make_payload(data: dict, minutes: int) -> dict:
    now = datetime.now(timezone.utc)
    return {
        **data,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }

def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    """
    Create a JWT access token.
    
    Generates a signed JWT token with expiration time. The 'sub' (subject)
    field is automatically converted to string if present.
    
    Args:
        data (dict): Token payload data (typically {"sub": user_id})
        expires_minutes (Optional[int]): Token expiration in minutes.
            Default: ACCESS_TOKEN_EXPIRE_MINUTES from config
    
    Returns:
        str: Encoded JWT token
    
    Example:
        >>> token = create_access_token({"sub": "user123"})
        >>> print(token[:20])
        'eyJhbGciOiJIUzI1NiIs'
        >>> 
        >>> # Custom expiration
        >>> short_token = create_access_token({"sub": "user123"}, expires_minutes=15)
    """
    to_encode = data.copy()
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])
    payload = _make_payload(to_encode, expires_minutes or ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    
    Args:
        token (str): JWT token to decode
    
    Returns:
        dict: Decoded token payload
    
    Raises:
        JWTError: If token is invalid or expired
    
    Example:
        >>> token = create_access_token({"sub": "user123"})
        >>> payload = decode_token(token)
        >>> print(payload["sub"])
        'user123'
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

# ---------- Cookies ----------
def set_login_cookie(response: Response, token: str):
    """
    Set authentication cookie in response.
    
    Configures a secure HttpOnly cookie with the JWT token,
    using settings from config (secure, samesite, max_age).
    
    Args:
        response (Response): FastAPI response object
        token (str): JWT token to store in cookie
    
    Example:
        >>> from fastapi import Response
        >>> response = Response()
        >>> token = create_access_token({"sub": "user123"})
        >>> set_login_cookie(response, token)
    """
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )

def clear_login_cookie(response: Response):
    """
    Clear authentication cookie from response.
    
    Removes the authentication cookie, effectively logging out the user.
    
    Args:
        response (Response): FastAPI response object
    
    Example:
        >>> from fastapi import Response
        >>> response = Response()
        >>> clear_login_cookie(response)
    """
    response.delete_cookie(COOKIE_NAME, path="/")

def _get_token_from_request(request: Request) -> Optional[str]:
    # Prefer Authorization header; fallback to cookie
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    cookie = request.cookies.get(COOKIE_NAME)
    return cookie

# ---------- Current user (+ sliding refresh) ----------
REFRESH_THRESHOLD_MIN = 60  # if < 60 min remaining, refresh

def _maybe_refresh(response: Response, payload: dict) -> None:
    exp = payload.get("exp")
    if not isinstance(exp, int):
        return
    now = int(datetime.now(timezone.utc).timestamp())
    remaining = exp - now
    if remaining < REFRESH_THRESHOLD_MIN * 60:
        # slide the session
        new_token = create_access_token({"sub": payload["sub"]})
        set_login_cookie(response, new_token)

def get_current_user(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    FastAPI dependency for getting the current authenticated user.
    
    Extracts and validates the JWT token from request headers or cookies,
    retrieves the user from database, and implements sliding session refresh
    (automatically extends token if expiring soon).
    
    Args:
        request (Request): FastAPI request object
        response (Response): FastAPI response object (for token refresh)
        db (Session): Database session (injected via Depends)
    
    Returns:
        User: Authenticated user object from database
    
    Raises:
        HTTPException: 401 if not authenticated or token invalid
        HTTPException: 404 if user not found in database
    
    Example:
        >>> from fastapi import Depends
        >>> from neurosurfer.server.security import get_current_user
        >>> 
        >>> @app.get("/me")
        >>> def get_profile(user = Depends(get_current_user)):
        ...     return {"username": user.username, "email": user.email}
    """
    token = _get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # Sliding session refresh (Response is guaranteed injected)
    _maybe_refresh(response, payload)

    # Stash for later use
    request.state.user = user
    return user


async def maybe_current_user(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Best-effort user resolver:
      - If Authorization bearer token exists and is valid -> returns User and sets request.state.user
      - If token missing/invalid -> returns None (NO exception)
    """
    token = _get_token_from_request(request)
    if not token:
        return None
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        _maybe_refresh(response, payload)
        request.state.user = user
        return user
    except JWTError:
        # Treat as anonymous
        return None


def resolve_actor_id(req: Request, user: Optional[object]) -> int:
    if user and getattr(user, "id", None) is not None:
        return int(user.id)
    hdr = req.headers.get("X-Actor-Id")
    if hdr and hdr.isdigit():
        return int(hdr)
    qp = req.query_params.get("aid")
    if qp and qp.isdigit():
        return int(qp)
    return 0  # anonymous