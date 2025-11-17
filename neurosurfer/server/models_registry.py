"""
Model Registry Module
=====================

This module provides a registry system for managing AI models in the server.
The ModelRegistry tracks available models with their metadata, enabling
model discovery, validation, and selection.

The registry stores model cards containing:
    - Model ID and family
    - Provider information
    - Context length
    - Description

Example:
    >>> from neurosurfer.server.models_registry import ModelRegistry
    >>> 
    >>> registry = ModelRegistry()
    >>> 
    >>> # Register models
    >>> registry.add(
    ...     id="gpt-4",
    ...     family="GPT",
    ...     provider="OpenAI",
    ...     context_length=8192,
    ...     description="GPT-4 model"
    ... )
    >>> 
    >>> # Check if model exists
    >>> if registry.exists("gpt-4"):
    ...     model = registry.get("gpt-4")
    ...     print(model.context_length)
    8192
    >>> 
    >>> # List all models
    >>> all_models = registry.all()
"""
from typing import Dict
from typing import List, Optional, Dict, Union
from .schemas.model_registry import ModelCard
from neurosurfer.models.chat_models import BaseChatModel as BaseChatModel
from dataclasses import dataclass

@dataclass
class ModelInfo:
    model_card: ModelCard
    llm: BaseChatModel

class ModelRegistry:
    """
    Registry for managing AI models and their metadata.
    
    This class provides a centralized registry for tracking available models,
    their capabilities, and metadata. Used by the server to validate model
    requests and provide model information to clients.
    
    Attributes:
        _models (Dict[str, ModelCard]): Internal registry mapping model IDs to model cards
    
    Example:
        >>> registry = ModelRegistry()
        >>> 
        >>> # Add models
        >>> registry.add(
        ...     id="llama-3-8b",
        ...     family="Llama",
        ...     provider="Meta",
        ...     context_length=8192,
        ...     description="Llama 3 8B parameter model"
        ... )
        >>> 
        >>> # Retrieve model info
        >>> model = registry.get("llama-3-8b")
        >>> print(f"{model.family} - {model.context_length} tokens")
        Llama - 8192 tokens
        >>> 
        >>> # Check availability
        >>> registry.exists("gpt-4")
        False
    """
    def __init__(self):
        """Initialize an empty model registry."""
        # self._models: Dict[str, ModelCard] = {}
        self._models: Dict[str, ModelInfo] = {}

    def add(
        self,
        llm: BaseChatModel,
        family: str = "Unknown",
        provider: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Register a new model in the registry.
        
        Args:
            id (str): Unique model identifier
            family (str): Model family/architecture. Default: "Unknown"
            provider (Optional[str]): Model provider (e.g., "OpenAI", "Meta"). Default: None
            context_length (int): Maximum context window size in tokens. Default: None
            description (Optional[str]): Human-readable model description. Default: None
        
        Raises:
            ValueError: If model ID is already registered
        
        Example:
            >>> registry.add(
            ...     id="gpt-3.5-turbo",
            ...     family="GPT",
            ...     provider="OpenAI",
            ...     context_length=4096,
            ...     description="Fast and efficient GPT-3.5 model"
            ... )
        """
        model_card = ModelCard(
            id=llm.model_name.split("/")[-1],
            family=family,
            provider=provider,
            context_length=llm.max_seq_length,
            description=description,
        )
        if model_card.id in self._models:
            raise ValueError(f"Model '{model_card.id}' is already registered.")
        self._models[model_card.id] = ModelInfo(model_card=model_card, llm=llm)

    def get_first_available(self) -> ModelInfo:
        """
        Retrieve the first available model from the registry.
        
        Returns:
            ModelInfo: First available model in the registry
        
        Raises:
            ValueError: If no models are registered
        
        Example:
            >>> model = registry.get_first_available()
            >>> print(model.model_card.id)
            gpt-4
        """
        if not self._models:
            raise ValueError("No models registered in the registry.")
        return next(iter(self._models.values()))

    def get(self, model_id: str) -> ModelInfo:
        """
        Retrieve model card by ID.
        
        Args:
            model_id (str): Model identifier to look up
        
        Returns:
            ModelCard: Model card containing model metadata
        
        Raises:
            KeyError: If model ID is not found in registry
        
        Example:
            >>> model = registry.get("gpt-4")
            >>> print(model.context_length)
            8192
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in registry.")
        return self._models[model_id]

    def exists(self, model_id: str) -> bool:
        """
        Check if a model is registered.
        
        Args:
            model_id (str): Model identifier to check
        
        Returns:
            bool: True if model exists in registry, False otherwise
        
        Example:
            >>> registry.exists("gpt-4")
            True
            >>> registry.exists("unknown-model")
            False
        """
        return model_id in self._models

    def all(self) -> Dict[str, ModelCard]:
        """
        Get all registered models.
        
        Returns:
            Dict[str, ModelCard]: Dictionary mapping model IDs to model cards
        
        Example:
            >>> all_models = registry.all()
            >>> for model_id, model_card in all_models.items():
            ...     print(f"{model_id}: {model_card.family}")
            gpt-4: GPT
            llama-3-8b: Llama
        """
        return {mid: mc.model_card for mid, mc in self._models.items()}
