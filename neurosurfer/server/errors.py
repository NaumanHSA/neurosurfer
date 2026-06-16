from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class OpenAIHTTPError(Exception):
    status_code: int
    message: str
    error_type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None

    def to_openai_error(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }
