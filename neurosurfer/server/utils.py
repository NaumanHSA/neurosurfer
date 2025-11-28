from typing import Dict, Any
from pathlib import Path
import os
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