"""
Base Tool Module
================

This module provides the base classes and data structures for the tool system
in Neurosurfer. Tools extend agent capabilities by providing specific functionalities
like SQL query execution, document generation, RAG queries, etc.

The module includes:
    - ToolResponse: Dataclass for structured tool outputs
    - BaseTool: Abstract base class for all tool implementations

All tools must inherit from BaseTool and implement the __call__ method.
Tools are registered in a Toolkit and can be invoked by agents (e.g., ReActAgent).
"""
from abc import ABC, abstractmethod
from typing import Any, Generator, Union
from dataclasses import dataclass, field
from .tool_spec import ToolSpec
from ..server.schemas import ChatCompletionChunk, ChatCompletionResponse


@dataclass
class ToolResponse:
    """
    Structured response from tool execution.
    
    This dataclass encapsulates the output of a tool, including whether it
    represents a final answer, the observation/result, and any extra data
    to pass to subsequent tool calls.
    
    Attributes:
        final_answer (bool): If True, the observation should be treated as the
            final answer to the user's query (no further tool calls needed)
        observation (Union[str, Generator]): The tool's output. Can be a string
            or a generator for streaming responses (ChatCompletionChunk style)
        extras (dict): Additional data to store in agent memory for subsequent
            tool calls. Default: {}
    
    Example:
        >>> # Simple tool response
        >>> response = ToolResponse(
        ...     final_answer=False,
        ...     observation="Found 3 matching records",
        ...     extras={"record_ids": [1, 2, 3]}
        ... )
        >>> 
        >>> # Final answer response
        >>> response = ToolResponse(
        ...     final_answer=True,
        ...     observation="The answer is 42"
        ... )
    """
    final_answer: bool
    observation: Union[str, ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]
    extras: dict = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for all tools in Neurosurfer.
    
    Tools extend agent capabilities by providing specific functionalities.
    Each tool must define a ToolSpec that describes its name, description,
    and input parameters. The spec is used by agents to understand how to
    invoke the tool.
    
    Attributes:
        spec (ToolSpec): Tool specification defining name, description, and inputs
    
    Abstract Methods:
        __call__(): Execute the tool with provided inputs
    
    Example:
        >>> from neurosurfer.tools import BaseTool, ToolResponse
        >>> from neurosurfer.tools.tool_spec import ToolSpec, InputParam
        >>> 
        >>> class MyTool(BaseTool):
        ...     spec = ToolSpec(
        ...         name="my_tool",
        ...         description="Does something useful",
        ...         inputs=[
        ...             InputParam(name="query", type="string", description="Input query", required=True)
        ...         ]
        ...     )
        ...     
        ...     def __call__(self, query: str, **kwargs):
        ...         result = f"Processed: {query}"
        ...         return ToolResponse(final_answer=False, observation=result)
        >>> 
        >>> tool = MyTool()
        >>> response = tool(query="test")
    """
    spec: ToolSpec

    def __init__(self) -> None:
        """
        Initialize the tool and validate its specification.
        
        Raises:
            TypeError: If the tool doesn't define a valid ToolSpec
        """
        if not hasattr(self, "spec") or not isinstance(self.spec, ToolSpec):
            raise TypeError(f"{self.__class__.__name__} must define a ToolSpec 'spec'.")
        self.spec.validate()


    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ToolResponse:
        """
        Execute the tool with the provided inputs.
        
        This method must be implemented by all concrete tool classes.
        It receives validated inputs from the agent and returns a ToolResponse.
        
        Args:
            *args: Positional arguments (typically not used)
            **kwargs: Tool inputs as defined in the ToolSpec, plus runtime context
                (e.g., llm, db_engine, embedder) injected by the agent
        
        Returns:
            ToolResponse: Structured output containing:
                - final_answer: Whether this is the final answer
                - observation: The tool's result (string or generator)
                - extras: Additional data for agent memory
        
        Raises:
            NotImplementedError: If not implemented by subclass
        
        Example:
            >>> def __call__(self, query: str, llm=None, **kwargs):
            ...     # Use injected runtime context
            ...     result = llm.ask(query) if llm else "No LLM available"
            ...     return ToolResponse(final_answer=True, observation=result)
        """
        raise NotImplementedError("Tool must implement __call__(**kwargs).")

    def get_tool_description(self) -> str:
        """
        Generate a formatted description of the tool.
        
        Returns:
            str: Formatted tool description in markdown
        """
        if not self.spec:
            raise ValueError("Tool must have a valid ToolSpec.")

        description = []
        description.append(f"Tool Name: `{self.spec.name}`")
        description.append(f"Description: {self.spec.description}")
        description.append(f"When to use: {self.spec.when_to_use}")
        description.append("Tool Inputs:")
        for p in self.spec.inputs:
            req = "required" if p.required else "optional"
            description.append(f"- `{p.name}`: {p.type} ({req}) — {p.description}")
        description.append(f"Tool Return: {self.spec.returns.type} — {self.spec.returns.description}\n")
        return "\n".join(description)
