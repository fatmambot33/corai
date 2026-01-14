"""Tool handler utilities for OpenAI SDK interactions.

This module provides generic tool handling infrastructure including argument
parsing, Pydantic validation, function execution, and result serialization.
These utilities reduce boilerplate and ensure consistent tool behavior.

Also provides declarative tool specification helpers for building tool
definitions from named metadata structures.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeAlias, TypeVar

from pydantic import BaseModel, ValidationError

from openai_sdk_helpers.response.tool_call import parse_tool_arguments
from openai_sdk_helpers.structure.base import StructureBase
from openai_sdk_helpers.utils import coerce_jsonable, customJSONEncoder
import json

T = TypeVar("T", bound=BaseModel)
StructureType: TypeAlias = type[StructureBase]


def serialize_tool_result(result: Any) -> str:
    """Serialize tool results into a standardized JSON string.

    Handles Pydantic models, lists, dicts, and plain strings with consistent
    JSON formatting. Pydantic models are serialized using model_dump(),
    while other types are converted to JSON or string representation.

    Parameters
    ----------
    result : Any
        Tool result to serialize. Can be a Pydantic model, list, dict, str,
        or any JSON-serializable type.

    Returns
    -------
    str
        JSON-formatted string representation of the result.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class Result(BaseModel):
    ...     value: int
    >>> serialize_tool_result(Result(value=42))
    '{"value": 42}'

    >>> serialize_tool_result(["item1", "item2"])
    '["item1", "item2"]'

    >>> serialize_tool_result("plain text")
    '"plain text"'

    >>> serialize_tool_result({"key": "value"})
    '{"key": "value"}'
    """
    if isinstance(result, BaseModel):
        return result.model_dump_json()

    payload = coerce_jsonable(result)
    return json.dumps(payload, cls=customJSONEncoder)


def tool_handler_factory(
    func: Callable[..., Any],
    *,
    input_model: type[T] | None = None,
) -> Callable[[Any], str]:
    """Create a generic tool handler that parses, validates, and serializes.

    Wraps a tool function with automatic argument parsing, optional Pydantic
    validation, execution, and result serialization. This eliminates
    repetitive boilerplate for tool implementations.

    The returned handler:
    1. Parses tool_call.arguments using parse_tool_arguments
    2. Validates arguments with input_model if provided
    3. Calls func with validated/parsed arguments (handles both sync and async)
    4. Serializes the result using serialize_tool_result

    Parameters
    ----------
    func : Callable[..., Any]
        The actual tool implementation function. Should accept keyword
        arguments matching the tool's parameter schema. Can be synchronous
        or asynchronous.
    input_model : type[BaseModel] or None, default None
        Optional Pydantic model for input validation. When provided,
        arguments are validated and converted to this model before being
        passed to func.

    Returns
    -------
    Callable[[Any], str]
        Handler function that accepts a tool_call object (with arguments
        and name attributes) and returns a JSON string result.

    Raises
    ------
    ValidationError
        If input_model is provided and validation fails.
    ValueError
        If argument parsing fails.

    Examples
    --------
    Basic usage without validation:

    >>> def search_tool(query: str, limit: int = 10):
    ...     return {"results": [f"Result for {query}"]}
    >>> handler = tool_handler_factory(search_tool)

    With Pydantic validation:

    >>> from pydantic import BaseModel
    >>> class SearchInput(BaseModel):
    ...     query: str
    ...     limit: int = 10
    >>> def search_tool(query: str, limit: int = 10):
    ...     return {"results": [f"Result for {query}"]}
    >>> handler = tool_handler_factory(search_tool, input_model=SearchInput)

    With async function:

    >>> async def async_search_tool(query: str, limit: int = 10):
    ...     return {"results": [f"Result for {query}"]}
    >>> handler = tool_handler_factory(async_search_tool)

    The handler can then be used with OpenAI tool calls:

    >>> class ToolCall:
    ...     def __init__(self):
    ...         self.arguments = '{"query": "test", "limit": 5}'
    ...         self.name = "search"
    >>> tool_call = ToolCall()
    >>> result = handler(tool_call)  # doctest: +SKIP
    """
    is_async = inspect.iscoroutinefunction(func)

    def handler(tool_call: Any) -> str:
        """Handle tool execution with parsing, validation, and serialization.

        Parameters
        ----------
        tool_call : Any
            Tool call object with 'arguments' and 'name' attributes.

        Returns
        -------
        str
            JSON-formatted result from the tool function.

        Raises
        ------
        ValueError
            If argument parsing fails.
        ValidationError
            If Pydantic validation fails (when input_model is provided).
        """
        # Extract tool name for error context (required)
        tool_name = getattr(tool_call, "name", "unknown")

        # Parse arguments with error context
        parsed_args = parse_tool_arguments(tool_call.arguments, tool_name=tool_name)

        # Validate with Pydantic if model provided
        if input_model is not None:
            validated_input = input_model(**parsed_args)
            # Convert back to dict for function call
            call_kwargs = validated_input.model_dump()
        else:
            call_kwargs = parsed_args

        # Execute function (sync or async with event loop detection)
        if is_async:
            # Handle async function with proper event loop detection
            try:
                loop = asyncio.get_running_loop()
                # We're inside an event loop, need to run in thread
                result_holder: dict[str, Any] = {"value": None, "exception": None}

                def _thread_func() -> None:
                    try:
                        result_holder["value"] = asyncio.run(func(**call_kwargs))
                    except Exception as exc:
                        result_holder["exception"] = exc

                thread = threading.Thread(target=_thread_func)
                thread.start()
                thread.join()

                if result_holder["exception"]:
                    raise result_holder["exception"]
                result = result_holder["value"]
            except RuntimeError:
                # No event loop running, can use asyncio.run directly
                result = asyncio.run(func(**call_kwargs))
        else:
            result = func(**call_kwargs)

        # Serialize result
        return serialize_tool_result(result)

    return handler


@dataclass(frozen=True)
class ToolSpec:
    """Capture tool metadata for response configuration.

    Provides a named structure for representing tool specifications, making
    tool definitions explicit and eliminating ambiguous tuple ordering.

    Supports tools with separate input and output structures, where the input
    structure defines the tool's parameter schema and the output structure
    documents the expected return type (for reference only).

    Attributes
    ----------
    structure : StructureType
        The StructureBase class that defines the tool's input parameter schema.
        Used to generate the OpenAI tool definition.
    tool_name : str
        Name identifier for the tool.
    tool_description : str
        Human-readable description of what the tool does.
    output_structure : StructureType or None, default=None
        Optional StructureBase class that defines the tool's output schema.
        This is for documentation/reference only and is not sent to OpenAI.
        Useful when a tool accepts one type of input but returns a different
        structured output.

    Examples
    --------
    Define a tool with same input/output structure:

    >>> from openai_sdk_helpers import ToolSpec
    >>> from openai_sdk_helpers.structure import PromptStructure
    >>> spec = ToolSpec(
    ...     structure=PromptStructure,
    ...     tool_name="web_agent",
    ...     tool_description="Run a web research workflow"
    ... )

    Define a tool with different input and output structures:

    >>> from openai_sdk_helpers.structure import PromptStructure, SummaryStructure
    >>> spec = ToolSpec(
    ...     structure=PromptStructure,
    ...     tool_name="summarizer",
    ...     tool_description="Summarize the provided prompt",
    ...     output_structure=SummaryStructure
    ... )
    """

    tool_name: str
    tool_description: str
    input_structure: StructureType
    output_structure: StructureType | None = None


def build_tool_definitions(tool_specs: list[ToolSpec]) -> list[dict]:
    """Build tool definitions from named tool specs.

    Converts a list of ToolSpec objects into OpenAI-compatible tool
    definitions for use in response configurations. Each ToolSpec is
    transformed into a tool definition using the structure's
    response_tool_definition method.

    Parameters
    ----------
    tool_specs : list[ToolSpec]
        List of tool specifications to convert.

    Returns
    -------
    list[dict]
        List of tool definition dictionaries ready for OpenAI API.

    Examples
    --------
    Build multiple tool definitions:

    >>> from openai_sdk_helpers import ToolSpec, build_tool_definitions
    >>> from openai_sdk_helpers.structure import PromptStructure
    >>> tools = build_tool_definitions([
    ...     ToolSpec(
    ...         structure=PromptStructure,
    ...         tool_name="web_agent",
    ...         tool_description="Run a web research workflow"
    ...     ),
    ...     ToolSpec(
    ...         structure=PromptStructure,
    ...         tool_name="vector_agent",
    ...         tool_description="Run a vector search workflow"
    ...     ),
    ... ])
    """
    return [
        spec.input_structure.response_tool_definition(
            tool_name=spec.tool_name,
            tool_description=spec.tool_description,
        )
        for spec in tool_specs
    ]


__all__ = [
    "serialize_tool_result",
    "tool_handler_factory",
    "StructureType",
    "ToolSpec",
    "build_tool_definitions",
]
