from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

# ============ Auth ============

class User(BaseModel):
    id: str
    name: str
    email: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    user: User