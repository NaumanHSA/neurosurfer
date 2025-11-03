# BaseVectorDB

**Module:** `neurosurfer.vectorstores.base`

**Type:** Abstract Base Class

## Overview

The **BaseVectorDB** is the abstract base class for all vector database implementations in Neurosurfer.

**Required Abstract Methods:**

- `add_documents(docs: List[Doc])` - Add documents with embeddings
- `similarity_search(query_embedding, top_k, metadata_filter, similarity_threshold)` - Search for similar documents
- `count()` - Get document count
- `list_all_documents(metadata_filter)` - Retrieve all documents
- `delete_documents(ids)` - Delete specific documents
- `delete_collection()` - Delete the collection
- `clear_collection()` - Clear all documents

## Doc Dataclass

Documents are represented using the `Doc` dataclass:

```python
@dataclass
class Doc:
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Creating Custom Vector Stores

```python
from neurosurfer.vectorstores.base import BaseVectorDB, Doc
from typing import List, Dict, Optional, Tuple, Any

class CustomVectorStore(BaseVectorDB):
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        # Your initialization
    
    def add_documents(self, docs: List[Doc]):
        """Add documents with embeddings to the store."""
        # Your implementation
        pass
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[Doc, float]]:
        """Search for similar documents.
        
        Returns:
            List of (Doc, similarity_score) tuples, sorted by score descending.
        """
        # Your implementation
        pass
    
    def count(self) -> int:
        """Return total number of documents."""
        # Your implementation
        pass
    
    def list_all_documents(
        self, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Doc]:
        """Retrieve all documents, optionally filtered by metadata."""
        # Your implementation
        pass
    
    def delete_documents(self, ids: List[str]):
        """Delete specific documents by ID."""
        # Your implementation
        pass
    
    def clear_collection(self):
        """Clear all documents but keep collection."""
        # Your implementation
        pass
    
    def delete_collection(self):
        """Delete the entire collection."""
        # Your implementation
        pass
```

## See Also

- [ChromaVectorStore](chroma.md)
- [Vector Stores Index](index.md)
