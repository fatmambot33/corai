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

