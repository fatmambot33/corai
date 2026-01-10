"""Tool call representation and argument parsing.

This module provides data structures and utilities for managing tool calls
in OpenAI response conversations, including conversion to OpenAI API formats
and robust argument parsing.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from ..utils.json.data_class import DataclassJSONSerializable


@dataclass
class ResponseToolCall(DataclassJSONSerializable):
    """Container for tool call data in a conversation.

    Stores the complete information about a tool invocation including
    the call identifier, tool name, input arguments, and execution output.
    Can convert to OpenAI API format for use in subsequent requests.

    Attributes
    ----------
    call_id : str
        Unique identifier for this tool call.
    name : str
        Name of the tool that was invoked.
    arguments : str
        JSON string containing the arguments passed to the tool.
    output : str
        JSON string representing the result produced by the tool handler.

    Methods
    -------
    to_response_input_item_param()
        Convert to OpenAI API tool call format.
    """

    call_id: str
    name: str
    arguments: str
    output: str

    def to_response_input_item_param(
        self,
    ) -> tuple[ResponseFunctionToolCallParam, FunctionCallOutput]:
        """Convert stored data into OpenAI API tool call objects.

        Creates the function call parameter and corresponding output object
        required by the OpenAI API for tool interaction.

        Returns
        -------
        tuple[ResponseFunctionToolCallParam, FunctionCallOutput]
            A two-element tuple containing:
            - ResponseFunctionToolCallParam: The function call representation
            - FunctionCallOutput: The function output representation

        Examples
        --------
        >>> tool_call = ResponseToolCall(
        ...     call_id="call_123",
        ...     name="search",
        ...     arguments='{"query": "test"}',
        ...     output='{"results": []}'
        ... )
        >>> func_call, func_output = tool_call.to_response_input_item_param()
        """
        from typing import cast

        function_call = cast(
            ResponseFunctionToolCallParam,
            {
                "arguments": self.arguments,
                "call_id": self.call_id,
                "name": self.name,
                "type": "function_call",
            },
        )
        function_call_output = cast(
            FunctionCallOutput,
            {
                "call_id": self.call_id,
                "output": self.output,
                "type": "function_call_output",
            },
        )
        return function_call, function_call_output


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


def parse_tool_arguments(arguments: str, tool_name: str) -> dict:
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
    >>> parse_tool_arguments('{"key": "value"}', tool_name="search")
    {'key': 'value'}

    >>> parse_tool_arguments("{'key': 'value'}", tool_name="search")
    {'key': 'value'}

    >>> parse_tool_arguments('{"ExampleTool": {"arg": "value"}}', "ExampleTool")
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
