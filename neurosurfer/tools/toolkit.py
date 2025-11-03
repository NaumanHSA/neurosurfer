"""
Toolkit Module
==============

This module provides the Toolkit class for managing and organizing tools
in Neurosurfer. The Toolkit acts as a registry for tools, allowing agents
to discover and invoke available tools.

The Toolkit:
    - Registers tools with validation
    - Maintains tool specifications
    - Generates formatted tool descriptions for agents
    - Prevents duplicate registrations

Example:
    >>> from neurosurfer.tools import Toolkit
    >>> from neurosurfer.tools.sql import SQLQueryTool
    >>> 
    >>> toolkit = Toolkit()
    >>> toolkit.register_tool(SQLQueryTool())
    >>> 
    >>> # Get tool descriptions for agent
    >>> descriptions = toolkit.get_tools_description()
    >>> 
    >>> # Access registered tools
    >>> tool = toolkit.registry["sql_query"]
"""
import logging
from typing import Optional
from typing import Dict

from .base_tool import BaseTool
from .tool_spec import ToolSpec


class Toolkit:
    """
    Tool registry and manager for agents.
    
    This class manages a collection of tools, providing registration,
    validation, and description generation capabilities. Agents use
    the Toolkit to discover and invoke available tools.
    
    Attributes:
        logger (logging.Logger): Logger instance
        registry (Dict[str, BaseTool]): Mapping of tool names to tool instances
        specs (Dict[str, ToolSpec]): Mapping of tool names to tool specifications
    
    Example:
        >>> toolkit = Toolkit()
        >>> 
        >>> # Register tools
        >>> toolkit.register_tool(MyTool())
        >>> toolkit.register_tool(AnotherTool())
        >>> 
        >>> # Get formatted descriptions
        >>> desc = toolkit.get_tools_description()
        >>> 
        >>> # Access tools
        >>> tool = toolkit.registry["my_tool"]
        >>> response = tool(param="value")
    """
    def __init__(
        self,
        logger: Optional[logging.Logger] = logging.getLogger(__name__)
    ):
        """
        Initialize the toolkit.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance. Default: module logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.registry: Dict[str, BaseTool] = {}
        self.specs: Dict[str, ToolSpec] = {}

    def register_tool(self, tool: BaseTool):
        """
        Register a tool in the toolkit.
        
        Validates the tool type and ensures no duplicate registrations.
        Once registered, the tool becomes available to agents.
        
        Args:
            tool (BaseTool): Tool instance to register
        
        Raises:
            TypeError: If tool is not a BaseTool subclass
            ValueError: If tool name is already registered
        
        Example:
            >>> from neurosurfer.tools.sql import SQLQueryTool
            >>> toolkit = Toolkit()
            >>> toolkit.register_tool(SQLQueryTool())
            Registered tool: sql_query
        """
        # enforce type check
        tool_name = tool.spec.name
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Invalid tool type: {type(tool).__name__}. "
                f"Expected a subclass of BaseTool."
            )

        if tool_name in self.registry:
            raise ValueError(f"Tool '{tool_name}' is already registered.")

        self.registry[tool_name] = tool
        self.specs[tool_name] = tool.spec
        self.logger.info(f"Registered tool: {tool_name}")

    def get_tools_description(self) -> str:
        """
        Generate formatted descriptions of all registered tools.
        
        Creates a markdown-formatted string describing each tool's:
        - Name and description
        - When to use it
        - Input parameters (with types and requirements)
        - Return type and description
        
        This description is used by agents to understand available tools.
        
        Returns:
            str: Formatted tool descriptions in markdown
        
        Example:
            >>> toolkit = Toolkit()
            >>> toolkit.register_tool(MyTool())
            >>> desc = toolkit.get_tools_description()
            >>> print(desc)
            ### `my_tool`
            Does something useful
            **When to use**: When you need to do X
            **Inputs**:
            - `param1`: string (required) — Description of param1
            **Returns**: string — Description of return value
        """
        lines = []
        for name, spec in self.specs.items():
            lines.append(f"### `{spec.name}`\n{spec.description}")
            lines.append(f"**When to use**: {spec.when_to_use}")
            lines.append("**Inputs**:")
            for p in spec.inputs:
                req = "required" if p.required else "optional"
                lines.append(f"- `{p.name}`: {p.type} ({req}) — {p.description}")
            lines.append(f"**Returns**: {spec.returns.type} — {spec.returns.description}\n")
        return "\n".join(lines)
