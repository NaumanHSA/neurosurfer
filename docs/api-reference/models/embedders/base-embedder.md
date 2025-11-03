# BaseEmbedder

**Module:** `neurosurfer.models.embedders.base`  
**Type:** Abstract base class

## Overview

`BaseEmbedder` defines the contract for embedding backends in Neurosurfer. All embedder implementations inherit from this class so they can be passed interchangeably to RAG pipelines, vector stores, or custom components.

### Responsibilities

- Manage a shared `logger` instance
- Load any backend-specific model inside the subclass constructor
- Implement a single `embed(query, **kwargs)` method that accepts either one string or a list of strings and returns vectors as Python lists

## Creating a custom embedder

```python
from typing import List, Union
from neurosurfer.models.embedders.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def __init__(self):
        super().__init__()
        self.model = load_something()

    def embed(self, query: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        if isinstance(query, list):
            return [self._encode(text) for text in query]
        return self._encode(query)

    def _encode(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
```

Return plain Python lists so downstream code can serialise or cast to `numpy`/`torch` as needed.

## See also

- [`SentenceTransformerEmbedder`](sentence-transformer.md)
- [`LlamaCppEmbedder`](llamacpp-embedder.md)

*mkdocstrings output is temporarily disabled while import hooks are updated.*
