from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml
from pydantic import BaseModel

from .schema import GraphSpec


def _pydantic_from_dict(model: type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    if hasattr(model, "model_validate"):
        return model.model_validate(data)  # Pydantic v2
    return model.parse_obj(data)          # Pydantic v1


def load_graph_from_dict(data: Dict[str, Any]) -> GraphSpec:
    return _pydantic_from_dict(GraphSpec, data)


def load_graph(path: Union[str, Path]) -> GraphSpec:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        import json
        data = json.loads(text)
    return load_graph_from_dict(data)
