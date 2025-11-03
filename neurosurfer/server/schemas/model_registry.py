from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# ============ Models ============

class ModelCard(BaseModel):
    """Represents one registered model."""
    id: str
    family: str = Field(..., description="Underlying model family, e.g. 'qwen2.5', 'llama3'")
    provider: Optional[str] = Field(None, description="Backend provider: local, vllm, openai, etc.")
    context_length: int = Field(32768, description="Max context length (tokens)")
    description: Optional[str] = None

    class Config:
        extra = "ignore"  # ignore any unknown fields from config

class ModelList(BaseModel):
    data: List[ModelCard]
    object: str = "list"