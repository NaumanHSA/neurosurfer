from pathlib import Path
import os
from .config import APP_DATA_PATH

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
