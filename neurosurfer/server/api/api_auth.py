"""
Authentication API Module
==========================

This module provides FastAPI routes for user authentication and account management.

Endpoints:
    - POST /auth/register: Register a new user account
    - POST /auth/login: Authenticate user and create session
    - POST /auth/logout: Clear authentication session
    - GET /auth/me: Get current user profile
    - POST /auth/delete_account: Delete user account and all data

Features:
    - Email-based registration with password hashing
    - JWT token authentication with secure cookies
    - User profile management
    - Account deletion with password confirmation
    - Automatic chat thread cleanup on account deletion

Security:
    - Passwords hashed with bcrypt
    - JWT tokens with configurable expiration
    - HttpOnly secure cookies
    - Password verification for sensitive operations

Example:
    >>> # Register new user
    >>> POST /auth/register
    >>> {
    ...     "email": "user@example.com",
    ...     "password": "secure-password",
    ...     "full_name": "John Doe"
    ... }
    >>> 
    >>> # Login
    >>> POST /auth/login
    >>> {
    ...     "email": "user@example.com",
    ...     "password": "secure-password"
    ... }
    >>> # Returns: {"token": "...", "user": {...}}
"""
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from ..schemas import LoginResponse, User as UserSchema
from ..security import get_db, hash_password, verify_password, create_access_token, set_login_cookie, clear_login_cookie, get_current_user
from ..db.models import User, ChatThread, Message

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterBody(BaseModel):
    """Request body for user registration."""
    email: EmailStr
    password: str
    full_name: str | None = None

class LoginBody(BaseModel):
    """Request body for user login."""
    email: EmailStr
    password: str

class DeleteBody(BaseModel):
    """Request body for account deletion (requires password confirmation)."""
    password: str

def user_to_schema(u: User) -> UserSchema:
    """
    Convert database User model to API schema.
    
    Args:
        u (User): Database user model
    
    Returns:
        UserSchema: API user schema with id, name, and email
    """
    # Map DB user to legacy schema.User (id/name/email)
    name = u.full_name or (u.email.split("@")[0] if u.email else "User")
    return UserSchema(id=str(u.id), name=name, email=u.email)

@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
def register(body: RegisterBody, response: Response, db: Session = Depends(get_db)):
    """
    Register a new user account.
    
    Creates a new user with hashed password and generates an authentication token.
    Email must be unique.
    
    Args:
        body (RegisterBody): Registration data (email, password, optional full_name)
        response (Response): FastAPI response object
        db (Session): Database session
    
    Returns:
        UserSchema: Created user information
    
    Raises:
        HTTPException: 400 if email already registered
    
    Example:
        >>> POST /auth/register
        >>> {"email": "user@example.com", "password": "pass123", "full_name": "John"}
        >>> # Returns: {"id": "1", "name": "John", "email": "user@example.com"}
    """
    exists = db.query(User).filter(User.email == body.email).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=body.email, full_name=body.full_name, hashed_password=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.id})
    # set_login_cookie(response, token)
    return user_to_schema(user)

@router.post("/login", response_model=LoginResponse)
def login(body: LoginBody, response: Response, db: Session = Depends(get_db)):
    """
    Authenticate user and create session.
    
    Validates credentials and creates JWT token stored in secure cookie.
    
    Args:
        body (LoginBody): Login credentials (email, password)
        response (Response): FastAPI response object (for setting cookie)
        db (Session): Database session
    
    Returns:
        LoginResponse: Authentication token and user information
    
    Raises:
        HTTPException: 401 if user not found or password invalid
    
    Example:
        >>> POST /auth/login
        >>> {"email": "user@example.com", "password": "pass123"}
        >>> # Returns: {"token": "eyJ...", "user": {...}}
    """
    user = db.query(User).filter(User.email == body.email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token = create_access_token({"sub": user.id})
    set_login_cookie(response, token)
    return LoginResponse(token=token, user=user_to_schema(user))

@router.post("/logout")
def logout(response: Response):
    """
    Logout user by clearing authentication cookie.
    
    Args:
        response (Response): FastAPI response object
    
    Returns:
        dict: Success confirmation
    
    Example:
        >>> POST /auth/logout
        >>> # Returns: {"ok": true}
    """
    clear_login_cookie(response)
    return {"ok": True}

@router.get("/me", response_model=UserSchema)
def me(user: User = Depends(get_current_user)):
    """
    Get current authenticated user profile.
    
    Args:
        user (User): Current user (injected via dependency)
    
    Returns:
        UserSchema: Current user information
    
    Raises:
        HTTPException: 401 if not authenticated
    
    Example:
        >>> GET /auth/me
        >>> # Headers: Authorization: Bearer <token>
        >>> # Returns: {"id": "1", "name": "John", "email": "user@example.com"}
    """
    return user_to_schema(user)

@router.post("/delete_account", response_model=UserSchema)
def delete_account(body: DeleteBody, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Delete user account and all associated data.
    
    Requires password confirmation. Deletes all chat threads, messages,
    and the user account. This operation is irreversible.
    
    Args:
        body (DeleteBody): Password confirmation
        user (User): Current user (injected via dependency)
        db (Session): Database session
    
    Returns:
        UserSchema: Deleted user information
    
    Raises:
        HTTPException: 401 if password is incorrect
    
    Example:
        >>> POST /auth/delete_account
        >>> {"password": "pass123"}
        >>> # Returns: {"id": "1", "name": "John", "email": "user@example.com"}
    """
    # Delete user but first authorize by checking password
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    # then delete all user threads
    db.query(ChatThread).filter(ChatThread.user_id == user.id).delete()
    # finally delete user
    db.query(User).filter(User.id == user.id).delete()
    db.commit()
    return user_to_schema(user)

