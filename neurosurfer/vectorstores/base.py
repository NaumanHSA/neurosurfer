"""
Base Vector Store Module
=========================

This module provides the abstract base class and data structures for vector databases
in Neurosurfer. It defines a unified interface for storing and retrieving document
embeddings across different vector store backends (Chroma, in-memory, etc.).

The module includes:
    - Doc: Dataclass representing a document with text, embedding, and metadata
    - BaseVectorDB: Abstract base class for all vector store implementations

All vector store implementations must inherit from BaseVectorDB and implement
methods for adding documents, similarity search, and collection management.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
# --------------------------
# Data structures   
# --------------------------

@dataclass
class Doc:
    """
    Document data structure for vector stores.
    
    Represents a single document with its text content, embedding vector,
    and associated metadata. Used throughout Neurosurfer's RAG and retrieval systems.
    
    Attributes:
        id (str): Unique identifier for the document
        text (str): The actual text content of the document
        embedding (Optional[List[float]]): Vector embedding of the text. None if not yet embedded.
        metadata (Dict[str, Any]): Additional metadata (e.g., filename, source, chunk_idx, etc.)
    
    Example:
        >>> doc = Doc(
        ...     id="doc_123",
        ...     text="Machine learning is a subset of AI.",
        ...     embedding=[0.1, 0.2, 0.3, ...],
        ...     metadata={"source": "textbook.pdf", "page": 42}
        ... )
    """
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseVectorDB(ABC):
    """
    Abstract base class for all vector database implementations in Neurosurfer.
    
    This class defines a unified interface for vector stores, enabling seamless
    switching between different backends (Chroma, in-memory, etc.) without
    changing application code.
    
    Core Operations:
        - add_documents(): Store documents with embeddings
        - similarity_search(): Find similar documents using vector similarity
        - list_all_documents(): Retrieve all documents (with optional filtering)
        - delete_documents(): Remove specific documents
        - clear_collection(): Remove all documents from collection
        - delete_collection(): Delete the entire collection
    
    Abstract Methods:
        All methods must be implemented by concrete vector store classes.
    
    Example:
        >>> class MyVectorDB(BaseVectorDB):
        ...     def add_documents(self, docs):
        ...         # Implementation
        ...         pass
        ...     # ... implement other methods
        >>> 
        >>> vectordb = MyVectorDB()
        >>> vectordb.add_documents([doc1, doc2])
        >>> results = vectordb.similarity_search(query_embedding, top_k=5)
    """
    @abstractmethod
    def add_documents(self, docs: List[Doc]):
        """
        Add documents with embeddings to the vector store.
        
        This method stores documents along with their vector embeddings and metadata.
        Documents can be retrieved later using similarity_search().
        
        Args:
            docs (List[Doc]): List of Doc objects to add. Each Doc must have:
                - id: Unique identifier
                - text: Document text content
                - embedding: Vector embedding (List[float])
                - metadata: Optional metadata dict
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            >>> docs = [
            ...     Doc(id="1", text="Hello", embedding=[0.1, 0.2], metadata={"source": "file.txt"}),
            ...     Doc(id="2", text="World", embedding=[0.3, 0.4], metadata={"source": "file.txt"})
            ... ]
            >>> vectordb.add_documents(docs)
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[Doc, float]]:
        """
        Perform similarity search to find documents most similar to the query.
        
        Uses vector similarity (typically cosine similarity) to find and rank
        documents by their relevance to the query embedding.
        
        Args:
            query_embedding (List[float]): Query vector to search for
            top_k (int): Maximum number of results to return. Default: 5
            metadata_filter (Optional[Dict[str, Any]]): Filter results by metadata.
                Only documents matching all key-value pairs are returned. Default: None
            similarity_threshold (Optional[float]): Minimum similarity score (0.0-1.0).
                Documents below this threshold are excluded. Default: None
        
        Returns:
            List[Tuple[Doc, float]]: List of (document, similarity_score) tuples,
                sorted by similarity in descending order (most similar first)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            >>> query_emb = embedder.embed("machine learning")
            >>> results = vectordb.similarity_search(
            ...     query_embedding=query_emb,
            ...     top_k=10,
            ...     metadata_filter={"category": "AI"},
            ...     similarity_threshold=0.7
            ... )
            >>> for doc, score in results:
            ...     print(f"Score: {score:.3f} - {doc.text[:50]}")
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def count(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            int: Number of documents currently stored
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            >>> count = vectordb.count()
            >>> print(f"Collection contains {count} documents")
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def list_all_documents(
        self,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Doc]:
        """
        Retrieve all documents from the collection.
        
        Args:
            metadata_filter (Optional[Dict[str, Any]]): Filter by metadata.
                Only documents matching all key-value pairs are returned. Default: None
        
        Returns:
            List[Doc]: List of all documents (or filtered subset)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            >>> # Get all documents
            >>> all_docs = vectordb.list_all_documents()
            >>> 
            >>> # Get documents from specific source
            >>> filtered_docs = vectordb.list_all_documents(
            ...     metadata_filter={"source": "manual.pdf"}
            ... )
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def delete_documents(self, docs: List[Doc]):
        """
        Delete specific documents from the collection.
        
        Args:
            docs (List[Doc]): List of documents to delete (matched by ID)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            >>> docs_to_delete = vectordb.list_all_documents(
            ...     metadata_filter={"source": "old_file.txt"}
            ... )
            >>> vectordb.delete_documents(docs_to_delete)
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def delete_collection(self):
        """
        Delete the entire collection permanently.
        
        This removes the collection and all its documents from the vector store.
        The collection cannot be recovered after deletion.
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            >>> vectordb.delete_collection()
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def clear_collection(self):
        """
        Remove all documents from the collection.
        
        This clears the collection but keeps it available for new documents.
        Typically implemented by dropping and recreating the collection.
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            >>> vectordb.clear_collection()
            >>> vectordb.count()  # Returns 0
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _stable_id(self, doc: Doc) -> str:
        """
        Generate a stable, deterministic ID for a document.
        
        Creates a unique identifier based on content hash and chunk index.
        This ensures the same content always gets the same ID, preventing duplicates.
        
        Args:
            doc (Doc): Document to generate ID for
        
        Returns:
            str: Stable ID in format "hash[:32]:chunk_idx"
        
        Example:
            >>> doc = Doc(id="temp", text="Hello", metadata={"chunk_idx": 0})
            >>> stable_id = vectordb._stable_id(doc)
            >>> print(stable_id)  # e.g., "a591a6d40bf420404a011733cfb7b190:0"
        """
        # Prefer content_hash set by your ingestor; else derive from text
        h = doc.metadata.get("content_hash") or hashlib.sha256((doc.text or "").encode("utf-8")).hexdigest()
        # optional: add a per-source offset index if you have it:
        chunk_idx = str(doc.metadata.get("chunk_idx", "0"))
        return f"{h[:32]}:{chunk_idx}"