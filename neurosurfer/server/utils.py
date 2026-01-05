from typing import Dict, Any, Generator, Optional, List
from pathlib import Path    
import os
import logging
import uuid

from neurosurfer.models.chat_models import BaseChatModel

from .config import APP_DATA_PATH
from .db.db import SessionLocal
from .db.models import NSFile

# Application paths
class ApplicationPaths:
    @staticmethod
    def rag_storage_path(user_id: int, thread_id: int) -> Path:
        path = os.path.join(APP_DATA_PATH, str(user_id), str(thread_id), "rag-storage")
        os.makedirs(path, exist_ok=True)
        return Path(path)

    @staticmethod
    def user_storage_path(user_id: int) -> Path:
        path = os.path.join(APP_DATA_PATH, f"ns_users_{user_id}")
        os.makedirs(path, exist_ok=True)
        return Path(path)

    @staticmethod
    def thread_storage_path(user_id: int, thread_id: int) -> Path:
        path = os.path.join(APP_DATA_PATH, f"ns_users_{user_id}", f"ns_threads_{user_id}_{thread_id}")
        os.makedirs(path, exist_ok=True)
        return Path(path)

    @staticmethod
    def thread_files_storage_path(user_id: int, thread_id: int) -> Path:
        path = os.path.join(APP_DATA_PATH, f"ns_users_{user_id}", f"ns_threads_{user_id}_{thread_id}", "files")
        os.makedirs(path, exist_ok=True)
        return Path(path)


def build_files_context(user_id: int, thread_id: int) -> Dict[str, Dict[str, Any]]:
    db = SessionLocal()
    files = (
        db.query(NSFile)
        .filter(NSFile.user_id == user_id, NSFile.thread_id == thread_id)
        .all()
    )
    ctx = {}
    for f in files:
        ctx[f.filename] = {
            "path": f.stored_path,
            "mime": f.mime,
            "size": f.size,
        }
    db.close()
    return ctx

def stream_chat_completion(response: Generator[str, None, None], model_name: Optional[str] = None):
    call_id = str(uuid.uuid1())
    for chunk in response:
        yield BaseChatModel._delta_chunk(call_id=call_id, model=model_name or "local-gpt", content=chunk)
    yield BaseChatModel._stop_chunk(call_id=call_id, model=model_name or "local-gpt", finish_reason="stop")

def non_stream_chat_completion(response: str, model_name: Optional[str] = None, prompt_tokens: int = 0, completion_tokens: int = 0):
    call_id = str(uuid.uuid1())
    yield BaseChatModel._final_nonstream_response(
        call_id=call_id, 
        model=model_name or "local-gpt", 
        content=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )

def parse_csv_env(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or default

def validate_cors(origins: List[str], allow_credentials: bool, logger: logging.Logger):
    if allow_credentials and "*" in origins:
        raise ValueError("CORS: cannot use '*' in allow_origins when allow_credentials=True")
    # Optional: log what you ended up with
    logger.info(f"CORS allow_origins = {origins} (credentials={allow_credentials})")
