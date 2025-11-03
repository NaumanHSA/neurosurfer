from typing import Dict, Union, List, Set, Optional, Any
import socket
import os
import logging
from pathlib import Path
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


# ============================================================================
# BASE MODEL CONFIGURATION
# ============================================================================

class BaseModelConfig(BaseSettings):
    """
    Base configuration shared across all model types.
    
    All model configs inherit these common parameters.
    Can be extended with model-specific settings.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")
    
    # Common generation parameters
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_seq_length: int = Field(default=4096, description="Maximum sequence length for model context", gt=0)
    max_new_tokens: int = Field(default=2000, description="Maximum tokens to generate", gt=0)
    top_k: int = Field(default=4, description="Top-k sampling parameter", gt=0)
    load_in_4bit: bool = Field(default=True, description="Enable 4-bit quantization")
    enable_thinking: bool = Field(default=False, description="Enable thinking mode")
    stop_words: Optional[List[str]] = Field(default=None, description="List of stop sequences for generation")
    system_prompt: str = Field(default="You are a helpful assistant. Answer questions to the best of your ability.", description="Default system prompt")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    
    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        Convert config to dictionary for unpacking into model __init__.
        
        Args:
            exclude_none: Exclude fields with None values
            
        Returns:
            Dictionary of configuration parameters
        """
        return self.model_dump(exclude_none=exclude_none, exclude_unset=False)


# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

class AppConfig(BaseSettings):
    """
    General application configuration.
    
    Manages application-level settings like versioning, networking,
    file paths, and runtime behavior.
    """
    model_config = SettingsConfigDict(env_prefix="APP_", env_nested_delimiter="__", env_file=".env", case_sensitive=False, extra="ignore")
    
    # Version
    app_name: str = Field(default="Neurosurfer")
    dev_version: str = Field(default="1.0.0")
    prod_version: str = Field(default="1.0.0")
    description: str = Field(default="SQL Agent - Let the Agent answer your queries")
    
    # Host settings
    host_ip: str = Field(default="0.0.0.0")
    host_port: int = Field(default=8081, gt=0, le=65535)
    host_protocol: str = Field(default="http")
    worker_timeout: int = Field(default=300)   # Workers silent for more than this many seconds are killed and restarted
    logs_level: str = Field(default="info")
    cors_origins: List[str] = Field(default=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173", 
        "http://0.0.0.0:5173",
        "http://server-ip:5173"
    ])
    
    # UI settings
    ui_host: str = Field(default="0.0.0.0")
    ui_port: int = Field(default=5173, gt=0, le=65535)
    ui_protocol: str = Field(default="http")

    # Server settings
    reload: bool = Field(default=False)
    workers: int = Field(default=1)
    enable_docs: bool = Field(default=True)
    
    # Paths
    ui_path: str = Field(default="ui_build")
    temp_path: str = Field(default="temp")
    logs_path: str = Field(default="logs")
    database_path: str = Field(default="./db_storage")
    
    # Docker environment
    is_docker: bool = Field(default=False)
    
    def get_dynamic_host_ip(self) -> str:
        """Get interface IP dynamically"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.253.155.219", 58162))
                print(s.getsockname()[0], "-----")
                return str(s.getsockname()[0])
        except Exception:
            return self.host_ip
    
    @property
    def host_url(self) -> str:
        """Construct full host URL"""
        return f"{self.host_protocol}://{self.host_ip}:{self.host_port}"
    
    @property
    def vector_store_path(self) -> str:
        """Vector store storage path"""
        return f"{self.database_path}/codemind_chroma"
    
    @property
    def database_url(self) -> str:
        """SQLite database URL"""
        return f"sqlite:///{self.database_path}/codemind_sqlite.db"


class DatabaseConfig(BaseSettings):
    """
    External database connection configuration.
    
    For SQL Server or other external databases used by the SQL Agent.
    """
    model_config = SettingsConfigDict(env_prefix="DB_", env_nested_delimiter="__", env_file=".env", case_sensitive=False, extra="ignore")
    
    server: str = Field(default="localhost")
    database: str = Field(default="")
    username: str = Field(default="")
    password: str = Field(default="")
    driver: str = Field(default="ODBC Driver 17 for SQL Server")
    port: str = Field(default="1433")


class ChunkerConfig(BaseSettings):
    """
    Configuration for the GenericCodeChunker.
    Controls chunking sizes, overlaps, and fallback behavior
    for both line-based (code-friendly) and char-based (generic text) splitting.
    """
    # ----- Line-based chunking (good for code) -----
    fallback_chunk_size: int = Field(default=25)
    """Number of lines per chunk when using line-based splitting."""

    overlap_lines: int = Field(default=3)
    """Number of overlapping lines between consecutive line-based chunks to preserve context."""

    max_chunk_lines: int = Field(default=1000)
    """Hard safety cap on lines per chunk (prevents giant chunks for huge files)."""

    comment_block_threshold: int = Field(default=4)
    """Minimum number of consecutive comment-only lines to treat as a 'comment block'
    for possible filtering/skipping during chunking."""

    # ----- Character-based chunking (good for prose / unknown formats) -----
    char_chunk_size: int = Field(default=1000)
    """Number of characters per chunk when using char-based splitting."""

    char_overlap: int = Field(default=150)
    """Number of overlapping characters between consecutive char-based chunks."""

    # ----- Special-case formats -----
    readme_max_lines: int = Field(default=30)
    """Max number of lines per chunk for README/Markdown files."""

    json_chunk_size: int = Field(default=1000)
    """Max number of characters per JSON chunk (used when pretty-printing large JSONs)."""

    # ----- Fallback policy -----
    fallback_mode: str = Field(default="char")
    """What to do when file extension has no registered strategy:
       - "char": Always fall back to character-based chunking.
       - "line": Always fall back to line-based chunking.
       - "auto": Detect if content looks like code → line-based; else → char-based."""

    # Safety caps for custom handlers
    max_returned_chunks: int = Field(default=500)
    """Hard limit on number of chunks a handler may return (post-sanitize)."""
    max_total_output_chars: int = Field(default=1_000_000)
    """Hard limit on total chars across all chunks (post-sanitize)."""
    min_chunk_non_ws_chars: int = Field(default=1)
    """Drop chunks that have fewer than this many non-whitespace characters."""


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================

class Config:
    """
    Main configuration class aggregating all sub-configurations.
    
    This is the primary entry point for all application configuration.
    Provides structured access to app, database, model, and processing configs.
    
    Usage:
        # Initialize config (loads from .env automatically)
        config = Config()
        
        # Access app settings
        print(config.app.host_url)
        
        # Configure and instantiate a model
        config.model.unsloth.model_name = "custom/model"
        config.model.unsloth.max_seq_length = 16000
        model = UnslothModel(**config.model.unsloth.to_dict())
    
    Attributes:
        app: Application-level configuration
        database: External database configuration
        chunker: Chunker configuration
    """
    
    def __init__(self):
        self.app = AppConfig()
        self.base_model = BaseModelConfig()
        self.database = DatabaseConfig()
        self.chunker = ChunkerConfig()
        
        # Ensure required directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories if they don't exist"""
        os.makedirs(self.app.temp_path, exist_ok=True)
        os.makedirs(self.app.logs_path, exist_ok=True)
        os.makedirs(self.app.database_path, exist_ok=True)
        os.makedirs(self.app.vector_store_path, exist_ok=True)
        
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger instance.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.app.logs_level.upper())
        return logger


# ============================================================================
# SINGLETON INSTANCE & BACKWARD COMPATIBILITY
# ============================================================================

# Singleton instance - import this in your code
config = Config()
