"""
Server Runtime Module
=====================

This module provides runtime management for server operations, including
request context tracking and operation lifecycle management.

The runtime system enables:
    - Request context creation with unique operation IDs
    - Stop signal management for cancellable operations
    - Operation lifecycle tracking
    - Thread-safe operation management

Classes:
    - RequestContext: Container for request-specific data
    - OperationManager: Manages operation lifecycle and cancellation

Example:
    >>> from neurosurfer.server.runtime import op_manager
    >>> 
    >>> # Create operation context
    >>> ctx = op_manager.create()
    >>> print(ctx.op_id)  # e.g., "op_a1b2c3d4e5f6"
    >>> 
    >>> # Check stop signal in long-running operation
    >>> while not ctx.stop_event.is_set():
    ...     # Do work
    ...     pass
    >>> 
    >>> # Stop operation from another thread
    >>> op_manager.stop(ctx.op_id)
    >>> 
    >>> # Clean up when done
    >>> op_manager.done(ctx.op_id)
"""
import time, uuid, threading
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RequestContext:
    """
    Container for request-specific runtime context.
    
    Holds metadata and control signals for a single request/operation,
    enabling cancellation and context passing throughout the request lifecycle.
    
    Attributes:
        op_id (str): Unique operation identifier (e.g., "op_a1b2c3d4e5f6")
        stop_event (threading.Event): Event for signaling operation cancellation
        headers (dict): Request headers or additional metadata
    
    Example:
        >>> ctx = RequestContext(
        ...     op_id="op_123",
        ...     stop_event=threading.Event(),
        ...     headers={"user-agent": "client/1.0"}
        ... )
        >>> 
        >>> # Check if operation should stop
        >>> if ctx.stop_event.is_set():
        ...     return "Operation cancelled"
    """
    op_id: str
    stop_event: threading.Event
    headers: dict
    meta: dict
    user_id: Optional[str] = None

class OperationManager:
    """
    Manages operation lifecycle and cancellation signals.
    
    This class provides centralized management of active operations,
    allowing creation, cancellation, and cleanup of request contexts.
    Thread-safe for concurrent access.
    
    Attributes:
        _ops (Dict[str, threading.Event]): Mapping of operation IDs to stop events
    
    Example:
        >>> manager = OperationManager()
        >>> 
        >>> # Create new operation
        >>> ctx = manager.create()
        >>> 
        >>> # Stop from another thread
        >>> manager.stop(ctx.op_id)
        >>> 
        >>> # Clean up
        >>> manager.done(ctx.op_id)
    """
    def __init__(self):
        """Initialize the operation manager with empty operation registry."""
        self._ops: Dict[str, threading.Event] = {}

    def create(self) -> RequestContext:
        """
        Create a new operation context with unique ID.
        
        Generates a unique operation ID, creates a stop event, and
        registers the operation for lifecycle management.
        
        Returns:
            RequestContext: New request context with unique op_id and stop_event
        
        Example:
            >>> ctx = op_manager.create()
            >>> print(ctx.op_id)
            'op_a1b2c3d4e5f6'
            >>> ctx.stop_event.is_set()
            False
        """
        op_id = f"op_{uuid.uuid4().hex[:12]}"
        ev = threading.Event()
        self._ops[op_id] = ev
        return RequestContext(op_id=op_id, stop_event=ev, headers={}, meta={})

    def stop(self, op_id: str) -> bool:
        """
        Signal an operation to stop.
        
        Sets the stop event for the specified operation, allowing
        the operation to gracefully terminate.
        
        Args:
            op_id (str): Operation ID to stop
        
        Returns:
            bool: True if operation was found and stopped, False if not found
        
        Example:
            >>> ctx = op_manager.create()
            >>> # ... operation running ...
            >>> success = op_manager.stop(ctx.op_id)
            >>> print(success)
            True
            >>> ctx.stop_event.is_set()
            True
        """
        ev = self._ops.get(op_id)
        if not ev:
            return False
        ev.set()
        return True

    def done(self, op_id: str):
        """
        Clean up a completed operation.
        
        Removes the operation from the registry after completion.
        Safe to call even if operation doesn't exist.
        
        Args:
            op_id (str): Operation ID to clean up
        
        Example:
            >>> ctx = op_manager.create()
            >>> # ... operation completes ...
            >>> op_manager.done(ctx.op_id)
        """
        self._ops.pop(op_id, None)

op_manager = OperationManager()
