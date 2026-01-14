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
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

from openai_sdk_helpers.response.tool_call import parse_tool_arguments
from openai_sdk_helpers.structure.base import StructureBase
from openai_sdk_helpers.utils import customJSONEncoder

StructureType: TypeAlias = type[StructureBase]


def serialize_tool_result(result: Any, *, tool_spec: "ToolSpec") -> str:
    """Serialize tool results into a standardized JSON string.

    Handles structured outputs with consistent JSON formatting. Outputs are
    validated and serialized through the ToolSpec output structure.

    Parameters
    ----------
    result : Any
        Tool result to serialize. Can be a structure instance or a compatible
        mapping for validation.
    tool_spec : ToolSpec
        Tool specification describing the expected output structure. The output
        structure validates and serializes the result.

    Returns
    -------
    str
        JSON-formatted string representation of the result.

    Raises
    ------
    ValueError
        If the tool specification is missing an output structure.

    Examples
    --------
    >>> from openai_sdk_helpers import ToolSpec
    >>> from openai_sdk_helpers.structure import PromptStructure
    >>> spec = ToolSpec(
    ...     tool_name="echo",
    ...     tool_description="Echo a prompt",
    ...     input_structure=PromptStructure,
    ...     output_structure=PromptStructure,
    ... )
    >>> serialize_tool_result({"prompt": "hello"}, tool_spec=spec)
    '{"prompt": "hello"}'
    """
    if tool_spec.output_structure is None:
        raise ValueError("ToolSpec.output_structure must be set for serialization.")

    output_structure = tool_spec.output_structure
    payload = output_structure.model_validate(result).to_json()
    return json.dumps(payload, cls=customJSONEncoder)


def unserialize_tool_arguments(tool_call: Any, *, tool_spec: "ToolSpec") -> StructureBase:
    """Unserialize tool call arguments into a structured input instance.

    Parameters
    ----------
    tool_call : Any
        Tool call object with 'arguments' and 'name' attributes.
    tool_spec : ToolSpec
        Tool specification describing the expected input structure.

    Returns
    -------
    StructureBase
        Validated input structure instance.

    Raises
    ------
    ValueError
        If argument parsing fails.
    ValidationError
        If input validation fails.
    """
    tool_name = getattr(tool_call, "name", tool_spec.tool_name)
    parsed_args = parse_tool_arguments(tool_call.arguments, tool_name=tool_name)
    return tool_spec.input_structure.from_json(parsed_args)


def tool_handler_factory(
    func: Callable[..., Any],
    *,
    tool_spec: "ToolSpec",
) -> Callable[[Any], str]:
    """Create a generic tool handler that parses, validates, and serializes.

    Wraps a tool function with automatic argument parsing, structured
    validation, execution, and result serialization. This eliminates
    repetitive boilerplate for tool implementations.

    The returned handler:
    1. Parses tool_call.arguments using parse_tool_arguments
    2. Validates arguments with the input structure
    3. Calls func with structured input (handles both sync and async)
    4. Serializes the result using serialize_tool_result

    Parameters
    ----------
    func : Callable[..., Any]
        The actual tool implementation function. Should accept keyword
        arguments matching the tool's parameter schema. Can be synchronous
        or asynchronous.
    tool_spec : ToolSpec
        Tool specification describing input and output structures. When
        provided, input parsing uses the input structure and output
        serialization uses the output structure.

    Returns
    -------
    Callable[[Any], str]
        Handler function that accepts a tool_call object (with arguments
        and name attributes) and returns a JSON string result.

    Raises
    ------
    ValueError
        If argument parsing fails.
    ValidationError
        If input validation fails.

    Examples
    --------
    Basic usage with ToolSpec:

    >>> from openai_sdk_helpers import ToolSpec
    >>> from openai_sdk_helpers.structure import PromptStructure
    >>> def search_tool(prompt: PromptStructure):
    ...     return {"prompt": prompt.prompt}
    >>> handler = tool_handler_factory(
    ...     search_tool,
    ...     tool_spec=ToolSpec(
    ...         tool_name="search",
    ...         tool_description="Run a search query",
    ...         input_structure=PromptStructure,
    ...         output_structure=PromptStructure,
    ...     ),
    ... )

    With async function:

    >>> async def async_search_tool(prompt: PromptStructure):
    ...     return {"prompt": prompt.prompt}
    >>> handler = tool_handler_factory(
    ...     async_search_tool,
    ...     tool_spec=ToolSpec(
    ...         tool_name="async_search",
    ...         tool_description="Run an async search query",
    ...         input_structure=PromptStructure,
    ...         output_structure=PromptStructure,
    ...     ),
    ... )

    The handler can then be used with OpenAI tool calls:

    >>> class ToolCall:
    ...     def __init__(self):
    ...         self.arguments = '{"query": "test", "limit": 5}'
    ...         self.name = "search"
    >>> tool_call = ToolCall()
    >>> result = handler(tool_call)  # doctest: +SKIP
    """
    is_async = inspect.iscoroutinefunction(func)

    def _call_with_input(validated_input: StructureBase) -> Any:
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        if len(params) == 1:
            param = params[0]
            if param.annotation is tool_spec.input_structure:
                return func(validated_input)
        return func(**validated_input.model_dump())

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
            If input validation fails.
        """
        validated_input = unserialize_tool_arguments(tool_call, tool_spec=tool_spec)

        # Execute function (sync or async with event loop detection)
        if is_async:
            # Handle async function with proper event loop detection
            try:
                loop = asyncio.get_running_loop()
                # We're inside an event loop, need to run in thread
                result_holder: dict[str, Any] = {"value": None, "exception": None}

                def _thread_func() -> None:
                    try:
                        result_holder["value"] = asyncio.run(
                            _call_with_input(validated_input)
                        )
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
                result = asyncio.run(_call_with_input(validated_input))
        else:
            result = _call_with_input(validated_input)

        # Serialize result
        return serialize_tool_result(result, tool_spec=tool_spec)

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
    tool_name : str
        Name identifier for the tool.
    tool_description : str
        Human-readable description of what the tool does.
    input_structure : StructureType
        The StructureBase class that defines the tool's input parameter schema.
        Used to generate the OpenAI tool definition.
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
    ...     tool_name="web_agent",
    ...     tool_description="Run a web research workflow",
    ...     input_structure=PromptStructure,
    ...     output_structure=PromptStructure
    ... )

    Define a tool with different input and output structures:

    >>> from openai_sdk_helpers.structure import PromptStructure, SummaryStructure
    >>> spec = ToolSpec(
    ...     tool_name="summarizer",
    ...     tool_description="Summarize the provided prompt",
    ...     input_structure=PromptStructure,
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
    ...         tool_name="web_agent",
    ...         tool_description="Run a web research workflow",
    ...         input_structure=PromptStructure,
    ...         output_structure=PromptStructure
    ...     ),
    ...     ToolSpec(
    ...         tool_name="vector_agent",
    ...         tool_description="Run a vector search workflow",
    ...         input_structure=PromptStructure
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
    "unserialize_tool_arguments",
    "tool_handler_factory",
    "StructureType",
    "ToolSpec",
    "build_tool_definitions",
]
