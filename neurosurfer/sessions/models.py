"""SessionRecord — metadata for one REPL conversation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SessionStatus = Literal["active", "closed", "interrupted"]


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


class SessionRecord(BaseModel):
    id: str = Field(default_factory=_new_session_id)
    task: str
    title: str = ""
    status: SessionStatus = "active"
    cwd: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_preview: str = ""
    turn_count: int = 0
    message_count: int = 0
    artifacts: list[str] = Field(default_factory=list)
