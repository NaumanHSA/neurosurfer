"""Session store — durable REPL conversation records.

Each session is a conversation with a specific task agent, persisted so the
user can resume it across CLI launches (like ChatGPT / Claude.ai chat history).

Layout (~/.neurosurfer/sessions/):
  <task>/
    <session_id>.json        — SessionRecord metadata
    <session_id>.hist.json   — full message history (JSON array of Message dicts)
"""

from .models import SessionRecord
from .store import SessionStore

__all__ = ["SessionRecord", "SessionStore"]
