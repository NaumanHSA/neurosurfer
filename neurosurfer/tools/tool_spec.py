# tools/spec.py
"""
Tool Specification Module
==========================

This module defines the specification system for tools in Neurosurfer.
Tool specifications provide structured metadata about tools, including
their inputs, outputs, and usage guidelines.

The specification system enables:
    - Type validation for tool inputs
    - Automatic documentation generation
    - Agent understanding of tool capabilities
    - Runtime input validation and sanitization

Classes:
    - ToolParam: Specification for a single tool parameter
    - ToolReturn: Specification for tool return value
    - ToolSpec: Complete tool specification

Example:
    >>> from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
    >>> 
    >>> spec = ToolSpec(
    ...     name="calculator",
    ...     description="Performs arithmetic operations",
    ...     when_to_use="When you need to calculate numbers",
    ...     inputs=[
    ...         ToolParam(name="operation", type="string", description="Operation to perform", required=True),
    ...         ToolParam(name="a", type="number", description="First operand", required=True),
    ...         ToolParam(name="b", type="number", description="Second operand", required=True)
    ...     ],
    ...     returns=ToolReturn(type="number", description="Result of the operation")
    ... )
    >>> spec.validate()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json

SUPPORTED_TYPES = {"string", "integer", "number", "boolean", "array", "object"}

TYPE_CHECK = {
    "string": lambda v: isinstance(v, str),
    "str": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number":  lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array":   lambda v: isinstance(v, list),
    "object":  lambda v: isinstance(v, dict),
}

@dataclass
class ToolParam:
    """
    Specification for a tool input parameter.
    
    Defines the name, type, description, and requirement status of a
    single tool parameter. Used for validation and documentation.
    
    Attributes:
        name (str): Parameter name
        type (str): Parameter type (string, integer, number, boolean, array, object)
        description (str): Human-readable description of the parameter
        required (bool): Whether the parameter is required. Default: True
    
    Example:
        >>> param = ToolParam(
        ...     name="query",
        ...     type="string",
        ...     description="SQL query to execute",
        ...     required=True
        ... )
    """
    name: str
    type: str            # one of SUPPORTED_TYPES
    description: str
    required: bool = True

@dataclass
class ToolReturn:
    """
    Specification for a tool's return value.
    
    Defines the type and description of what a tool returns.
    
    Attributes:
        type (str): Return type (string, integer, number, boolean, array, object)
        description (str): Human-readable description of the return value
    
    Example:
        >>> ret = ToolReturn(
        ...     type="object",
        ...     description="Query results as a dictionary"
        ... )
    """
    type: str            # e.g. "string", "object"
    description: str

@dataclass
class ToolSpec:
    """
    Complete specification for a tool.
    
    Defines all metadata needed for a tool: name, description, usage guidelines,
    input parameters, and return type. Used for validation, documentation, and
    agent understanding.
    
    Attributes:
        name (str): Unique tool identifier
        description (str): Brief description of what the tool does
        when_to_use (str): Guidelines for when to use this tool
        inputs (List[ToolParam]): List of input parameter specifications
        returns (ToolReturn): Return value specification
    
    Methods:
        validate(): Validate the specification
        to_json(): Convert to JSON-serializable dict
        check_inputs(): Validate and sanitize runtime inputs
    
    Example:
        >>> spec = ToolSpec(
        ...     name="web_search",
        ...     description="Search the web for information",
        ...     when_to_use="When you need current information from the internet",
        ...     inputs=[
        ...         ToolParam(name="query", type="string", description="Search query", required=True),
        ...         ToolParam(name="max_results", type="integer", description="Max results", required=False)
        ...     ],
        ...     returns=ToolReturn(type="array", description="List of search results")
        ... )
        >>> spec.validate()
        >>> inputs = spec.check_inputs({"query": "AI news"})
    """
    name: str
    description: str
    when_to_use: str
    inputs: List[ToolParam]
    returns: ToolReturn

    def validate(self) -> None:
        """
        Validate the tool specification.
        
        Checks that:
        - Name, description, and when_to_use are non-empty
        - At least one input parameter is defined
        - All parameter types are supported
        - Parameter names are unique
        - Return type is supported
        
        Raises:
            ValueError: If validation fails
        
        Example:
            >>> spec = ToolSpec(...)
            >>> spec.validate()  # Raises ValueError if invalid
        """
        if not self.name or not self.description or not self.when_to_use:
            raise ValueError("ToolSpec must have name, description, when_to_use.")
        if not self.inputs:
            raise ValueError(f"ToolSpec({self.name}) must define at least one input param.")
        names = set()
        for p in self.inputs:
            if p.type not in SUPPORTED_TYPES:
                raise ValueError(f"{self.name}.{p.name} has unsupported type '{p.type}'.")
            if not p.name:
                raise ValueError(f"{self.name} has an input with empty name.")
            if p.name in names:
                raise ValueError(f"{self.name} has duplicate input '{p.name}'.")
            names.add(p.name)
        if self.returns.type not in SUPPORTED_TYPES:
            raise ValueError(f"{self.name}.returns has unsupported type '{self.returns.type}'.")

    def to_json(self) -> Dict[str, Any]:
        """
        Convert specification to JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the spec
        
        Example:
            >>> spec = ToolSpec(...)
            >>> json_data = spec.to_json()
            >>> print(json_data["name"])
            'my_tool'
        """
        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "inputs": [
                {"name": p.name, "type": p.type, "description": p.description, "required": p.required}
                for p in self.inputs
            ],
            "returns": {"type": self.returns.type, "description": self.returns.description},
        }

    def check_inputs(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize runtime inputs against the specification.
        
        Performs strict validation:
        - Ensures all required parameters are present
        - Rejects extra/unknown parameters
        - Validates parameter types
        
        Args:
            raw (Dict[str, Any]): Raw input dictionary from LLM or user
        
        Returns:
            Dict[str, Any]: Validated input dictionary (same as input if valid)
        
        Raises:
            ValueError: If validation fails (missing required, wrong type, extra params)
        
        Example:
            >>> spec = ToolSpec(
            ...     inputs=[ToolParam(name="x", type="number", description="X", required=True)],
            ...     ...
            ... )
            >>> validated = spec.check_inputs({"x": 42})
            >>> # Raises ValueError:
            >>> spec.check_inputs({"x": "not a number"})
            >>> spec.check_inputs({"y": 42})  # Missing required 'x'
            >>> spec.check_inputs({"x": 42, "z": 10})  # Extra param 'z'
        """
        # Required
        for p in self.inputs:
            if p.required and p.name not in raw:
                raise ValueError(f"Missing required input: {p.name}")

        # No extras
        allowed = {p.name for p in self.inputs}
        extras = set(raw.keys()) - allowed
        if extras:
            raise ValueError(f"Unexpected inputs for {self.name}: {sorted(extras)}")

        # Types
        for p in self.inputs:
            if p.name in raw:
                val = raw[p.name]
                if not TYPE_CHECK[p.type](val):
                    raise ValueError(f"Input '{p.name}' expected {p.type}, got {type(val).__name__}")
        return raw
