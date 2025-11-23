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
from typing import Optional, Dict, List, Any

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
        tools: List[BaseTool] = [],
        logger: Optional[logging.Logger] = logging.getLogger(__name__)
    ):
        """
        Initialize the toolkit.
        
        Args:
            tools (List[BaseTool]): List of tools to register. Default: empty list
            logger (Optional[logging.Logger]): Logger instance. Default: module logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.registry: Dict[str, BaseTool] = {}
        self.specs: Dict[str, ToolSpec] = {}
        for tool in tools:
            self.register_tool(tool)

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
            Available tools:
            Tool Name: <tool_name>
            Description: <tool_description>
            When to use: <when_to_use>
            Tool Inputs:
            - <input_name>: <input_type> (<required/optional>) — <input_description>
              ...
            Tool Return: <return_type> — <return_description>
            
        """
        tools_descriptions = ["\nAvailable tools:"]
        tools_descriptions.extend([t.get_tool_description() for t in self.registry.values()])
        return "\n".join(tools_descriptions).strip()


    def build_tool_args(
        tool_spec: ToolSpec,
        llm_args: Dict[str, Any],
        context: Dict[str, Any],   # graph_inputs, deps, etc.
        bindings: Optional[Dict[str, Any]] = None,  # optional per-node bindings
    ) -> Dict[str, Any]:
        """
        Merge LLM-provided args with system/graph-provided args
        based on ToolParam.llm flag and optional bindings.
        """
        bindings = bindings or {}
        final: Dict[str, Any] = {}

        for param in tool_spec.inputs:
            name = param.name

            # 1) If we have a binding for this param, that always wins
            if name in bindings:
                final[name] = resolve_binding(bindings[name], context)
                continue

            # 2) If param is LLM-driven, use whatever model gave us (if provided)
            if param.llm:
                if name in llm_args:
                    final[name] = llm_args[name]
                # if required and missing, you can raise or try repair
                continue

            # 3) Non-LLM param without explicit binding: try context
            #    (graph_inputs, dependencies, etc.)
            value = resolve_from_context(name, context)
            if value is not None:
                final[name] = value

        return final
