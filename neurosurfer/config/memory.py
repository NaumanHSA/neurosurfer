"""Long-term memory settings (Pillar 1).

``enabled`` toggles retrieval/injection and end-of-run distillation;
``embeddings_backend`` is "none"/"bm25" (default) or an optional embedder that
always degrades to BM25 on failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryConfig:
    dir: Path = field(default_factory=lambda: Path.home() / ".neurosurfer" / "memory")
    enabled: bool = True
    embeddings_backend: str = "none"
    token_budget: int = 1000
