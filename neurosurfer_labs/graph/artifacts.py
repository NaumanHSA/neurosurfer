# neurosurfer/agents/graph/artifacts.py
from typing import Any, Dict
from pathlib import Path
import json, uuid

class ArtifactStore:
    def put(self, obj: Any) -> str:
        raise NotImplementedError
    def get(self, key: str) -> Any:
        raise NotImplementedError

class LocalArtifactStore(ArtifactStore):
    """
    Stores JSON-serializable blobs and small files in a local folder.
    Non-JSON objects are stringified; adapt as needed.
    """
    def __init__(self, base_dir: str | Path = ".ns_artifacts"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def put(self, obj: Any) -> str:
        key = uuid.uuid4().hex
        path = self.base / f"{key}.json"
        try:
            path.write_text(json.dumps(obj, ensure_ascii=False, default=str))
        except Exception:
            path.write_text(json.dumps({"__string__": str(obj)}, ensure_ascii=False))
        return key

    def get(self, key: str) -> Any:
        path = self.base / f"{key}.json"
        if not path.exists():
            raise KeyError(f"Artifact not found: {key}")
        txt = path.read_text()
        try:
            return json.loads(txt)
        except Exception:
            return txt
