from __future__ import annotations

import json
from typing import Any


def sse_data(payload: Any) -> bytes:
    if isinstance(payload, (dict, list)):
        data = json.dumps(payload, ensure_ascii=False, default=str)
    else:
        data = str(payload)
    return f"data: {data}\n\n".encode("utf-8")


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def sse_ping() -> bytes:
    return b": ping\n\n"
