"""Named tool specifications for response configuration.

This module provides a structured approach to defining tool metadata and
building tool definitions, reducing boilerplate when creating response
configurations with multiple tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .base import BaseStructure

StructureType: TypeAlias = type[BaseStructure]


@dataclass(frozen=True)
class ToolSpec:
    """Capture tool metadata for response configuration.

    Provides a named structure for representing tool specifications, making
    tool definitions explicit and eliminating ambiguous tuple ordering.

    Attributes
    ----------
    structure : StructureType
        The BaseStructure class that defines the tool schema.
    tool_name : str
        Name identifier for the tool.
    tool_description : str
        Human-readable description of what the tool does.

    Examples
    --------
    Define a tool specification:

    >>> from openai_sdk_helpers.structure import ToolSpec, PromptStructure
    >>> spec = ToolSpec(
    ...     structure=PromptStructure,
    ...     tool_name="web_agent",
    ...     tool_description="Run a web research workflow"
    ... )
    """

    structure: StructureType
    tool_name: str
    tool_description: str


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

    >>> from openai_sdk_helpers.structure import (
    ...     ToolSpec,
    ...     build_tool_definitions,
    ...     PromptStructure
    ... )
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
        spec.structure.response_tool_definition(
            tool_name=spec.tool_name,
            tool_description=spec.tool_description,
        )
        for spec in tool_specs
    ]
