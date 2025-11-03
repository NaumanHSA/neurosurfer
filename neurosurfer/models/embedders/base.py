"""
Base Embedder Module
====================

This module provides the abstract base class for all embedding models in Neurosurfer.
It defines a unified interface for generating text embeddings from different backends
(Sentence Transformers, LlamaCpp, etc.).

All concrete embedder implementations must inherit from BaseEmbedder and implement
the embed() method to generate vector embeddings from text.
"""
from typing import List, Union, Optional
import logging
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models in Neurosurfer.
    
    This class provides a unified interface for generating text embeddings
    from various backends. Embeddings are used for semantic search, RAG,
    and similarity computations.
    
    Attributes:
        logger (logging.Logger): Logger instance for debugging
        model: The underlying embedding model instance (implementation-specific)
    
    Abstract Methods:
        embed(): Generate embeddings for single or multiple text inputs
    
    Example:
        >>> class MyEmbedder(BaseEmbedder):
        ...     def embed(self, query, **kwargs):
        ...         # Generate embeddings
        ...         return embeddings
        >>> 
        >>> embedder = MyEmbedder()
        >>> embedding = embedder.embed("Hello world")
        >>> embeddings = embedder.embed(["Text 1", "Text 2"])
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the base embedder.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance. If None, creates default logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
    
    @abstractmethod
    def embed(self, query: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text input(s).
        
        This method must be implemented by all concrete embedder classes.
        It should handle both single strings and lists of strings.
        
        Args:
            query (Union[str, List[str]]): Single text string or list of text strings
            **kwargs: Additional embedder-specific parameters (e.g., normalize_embeddings)
        
        Returns:
            Union[List[float], List[List[float]]]:
                - If query is str: Returns List[float] (single embedding vector)
                - If query is List[str]: Returns List[List[float]] (list of embedding vectors)
        
        Example:
            >>> # Single text
            >>> embedding = embedder.embed("Hello world")
            >>> len(embedding)  # e.g., 384 for all-MiniLM-L6-v2
            384
            
            >>> # Multiple texts
            >>> embeddings = embedder.embed(["Text 1", "Text 2"])
            >>> len(embeddings)
            2
        """
        pass

