"""Tool handler utilities for OpenAI SDK interactions.

This module provides generic tool handling infrastructure including argument
parsing, Pydantic validation, function execution, and result serialization.
These utilities reduce boilerplate and ensure consistent tool behavior.

Also provides declarative tool specification helpers for building tool
definitions from named metadata structures.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeAlias, get_type_hints

from openai_sdk_helpers.structure.base import StructureBase
from openai_sdk_helpers.utils import customJSONEncoder

StructureType: TypeAlias = type[StructureBase]
ToolHandler: TypeAlias = Callable[[Any], str | Any]


@dataclass(frozen=True)
class ToolHandlerRegistration:
    """Bundle a tool handler with optional ToolSpec metadata.

    Parameters
    ----------
    handler : ToolHandler
        Callable that executes the tool and returns a serializable payload.
    tool_spec : ToolSpec or None, default None
        Optional ToolSpec used to parse tool outputs based on the tool name.

    Attributes
    ----------
    handler : ToolHandler
        Callable that executes the tool and returns a serializable payload.
    tool_spec : ToolSpec
        ToolSpec describing the tool input/output structures.

    Methods
    -------
    __init__(handler, tool_spec)
        Initialize the registration with a handler and ToolSpec.
    """

    handler: ToolHandler
    tool_spec: ToolSpec


def _to_snake_case(name: str) -> str:
    """Convert a PascalCase or camelCase string to snake_case.

    Parameters
    ----------
    name : str
        The name to convert.

    Returns
    -------
    str
        The snake_case version of the name.

    Examples
    --------
    >>> _to_snake_case("ExampleStructure")
    'example_structure'
    >>> _to_snake_case("MyToolName")
    'my_tool_name'
    """
    # First regex: Insert underscore before uppercase letters followed by
    # lowercase letters (e.g., "Tool" in "ExampleTool" becomes "_Tool")
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Second regex: Insert underscore between lowercase/digit and uppercase
    # (e.g., "e3" followed by "T" becomes "e3_T")
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _unwrap_arguments(parsed: dict, tool_name: str) -> dict:
    """Unwrap arguments if wrapped in a single-key dict.

    Some responses wrap arguments under a key matching the structure class
    name (e.g., {"ExampleStructure": {...}}) or snake_case variant
    (e.g., {"example_structure": {...}}). This function detects and unwraps
    such wrappers to normalize the payload.

    Parameters
    ----------
    parsed : dict
        The parsed arguments dictionary.
    tool_name : str
        The tool name, used to match potential wrapper keys.

    Returns
    -------
    dict
        Unwrapped arguments dictionary, or original if no wrapper detected.

    Examples
    --------
    >>> _unwrap_arguments({"ExampleTool": {"arg": "value"}}, "ExampleTool")
    {'arg': 'value'}
    >>> _unwrap_arguments({"example_tool": {"arg": "value"}}, "ExampleTool")
    {'arg': 'value'}
    >>> _unwrap_arguments({"arg": "value"}, "ExampleTool")
    {'arg': 'value'}
    """
    # Only unwrap if dict has exactly one key
    if not isinstance(parsed, dict) or len(parsed) != 1:
        return parsed

    wrapper_key = next(iter(parsed))
    wrapped_value = parsed[wrapper_key]

    # Only unwrap if the value is also a dict
    if not isinstance(wrapped_value, dict):
        return parsed

    # Check if wrapper key matches tool name (case-insensitive or snake_case)
    tool_name_lower = tool_name.lower()
    tool_name_snake = _to_snake_case(tool_name)
    wrapper_key_lower = wrapper_key.lower()

    if wrapper_key_lower in (tool_name_lower, tool_name_snake):
        return wrapped_value

    return parsed


def _parse_tool_arguments(arguments: str, tool_name: str) -> dict:
    """Parse tool call arguments with fallback for malformed JSON.

    Attempts to parse arguments as JSON first, then falls back to
    ast.literal_eval for cases where the OpenAI API returns minor
    formatting issues like single quotes instead of double quotes.
    Provides clear error context including tool name and raw payload.

    Also handles unwrapping of arguments that are wrapped in a single-key
    dictionary matching the tool name (e.g., {"ExampleStructure": {...}}).

    Parameters
    ----------
    arguments : str
        Raw argument string from a tool call, expected to be JSON.
    tool_name : str
        Tool name for improved error context (required).

    Returns
    -------
    dict
        Parsed dictionary of tool arguments, with wrapper unwrapped if present.

    Raises
    ------
    ValueError
        If the arguments cannot be parsed as valid JSON or Python literal.
        Error message includes tool name and payload excerpt for debugging.

    Examples
    --------
    >>> _parse_tool_arguments('{"key": "value"}', tool_name="search")
    {'key': 'value'}

    >>> _parse_tool_arguments("{'key': 'value'}", tool_name="search")
    {'key': 'value'}

    >>> _parse_tool_arguments('{"ExampleTool": {"arg": "value"}}', "ExampleTool")
    {'arg': 'value'}
    """
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(arguments)
        except Exception as exc:  # noqa: BLE001
            # Build informative error message with context
            payload_preview = (
                arguments[:100] + "..." if len(arguments) > 100 else arguments
            )
            raise ValueError(
                f"Failed to parse tool arguments for tool '{tool_name}'. "
                f"Raw payload: {payload_preview}"
            ) from exc

    # Unwrap if wrapped in a single-key dict matching tool name
    return _unwrap_arguments(parsed, tool_name)


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
    1. Parses tool_call.arguments using ToolSpec.unserialize_tool_arguments
    2. Validates arguments with the input structure
    3. Calls func with structured input (handles both sync and async)
    4. Serializes the result using ToolSpec.serialize_tool_result

    Forward-referenced annotations on single-argument tool callables are
    resolved when possible to decide whether to pass the structured input.

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
            try:
                type_hints = get_type_hints(func)
            except (NameError, TypeError):
                type_hints = {}
            annotated_type = type_hints.get(param.name, param.annotation)
            if isinstance(annotated_type, str):
                if (
                    annotated_type == tool_spec.input_structure.__name__
                    or annotated_type.endswith(f".{tool_spec.input_structure.__name__}")
                ):
                    return func(validated_input)
            if annotated_type is tool_spec.input_structure or (
                inspect.isclass(annotated_type)
                and issubclass(annotated_type, StructureBase)
            ):
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
        validated_input = tool_spec.unserialize_arguments(tool_call)

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
        return tool_spec.serialize_tool_results(result)

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
    output_structure : StructureType
        StructureBase class that defines the tool's output schema.
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
    tool_description: str | None
    input_structure: StructureType
    output_structure: StructureType

    def __post_init__(self) -> None:
        """Validate required ToolSpec fields."""
        if self.output_structure is None:
            raise ValueError("ToolSpec.output_structure must be set.")

    def serialize_tool_results(self, tool_results: Any) -> str:
        """Serialize tool results into a standardized JSON string.

        Handles structured outputs with consistent JSON formatting. Outputs are
        validated and serialized through the ToolSpec output structure.

        Parameters
        ----------
        result : Any
            Tool result to serialize. Can be a structure instance or a compatible
            mapping for validation.

        Returns
        -------
        str
            JSON-formatted string representation of the result.

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
        >>> spec.serialize_tool_result({"prompt": "hello"})
        '{"prompt": "hello"}'
        """
        output_structure = self.output_structure
        payload = output_structure.model_validate(tool_results).to_json()
        return json.dumps(payload, cls=customJSONEncoder)

    def unserialize_arguments(self, tool_call: Any) -> StructureBase:
        """Unserialize tool call arguments into a structured input instance.

        Parameters
        ----------
        tool_call : Any
            Tool call object with 'arguments' and 'name' attributes.

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
        tool_name = getattr(tool_call, "name", self.tool_name)
        parsed_args = _parse_tool_arguments(tool_call.arguments, tool_name=tool_name)
        return self.input_structure.from_json(parsed_args)

    def as_tool_definition(self) -> dict:
        """Generate OpenAI-compatible tool definition from the ToolSpec.

        Uses the input structure to create a tool definition dictionary
        suitable for inclusion in OpenAI API calls.

        Returns
        -------
        dict
            Tool definition dictionary.

        Examples
        --------
        >>> from openai_sdk_helpers import ToolSpec
        >>> from openai_sdk_helpers.structure import PromptStructure
        >>> spec = ToolSpec(
        ...     tool_name="web_agent",
        ...     tool_description="Run a web research workflow",
        ...     input_structure=PromptStructure,
        ...     output_structure=PromptStructure
        ... )
        >>> spec.as_tool_definition()
        {
            "name": "web_agent",
            "description": "Run a web research workflow",
            "parameters": { ... }  # Schema from PromptStructure
        }
        """
        return self.input_structure.response_tool_definition(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
        )


def build_tool_definition_list(tool_specs: list[ToolSpec]) -> list[dict]:
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

    >>> from openai_sdk_helpers import ToolSpec, build_tool_definition_list
    >>> from openai_sdk_helpers.structure import PromptStructure
    >>> tools = build_tool_definition_list([
    ...     ToolSpec(
    ...         tool_name="web_agent",
    ...         tool_description="Run a web research workflow",
    ...         input_structure=PromptStructure,
    ...         output_structure=PromptStructure
    ...     ),
    ...     ToolSpec(
    ...         tool_name="vector_agent",
    ...         tool_description="Run a vector search workflow",
    ...         input_structure=PromptStructure,
    ...         output_structure=PromptStructure
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
    "tool_handler_factory",
    "StructureType",
    "ToolHandler",
    "ToolHandlerRegistration",
    "ToolSpec",
    "build_tool_definition_list",
]
