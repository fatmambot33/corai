"""Tests for ToolSpec and build_tool_definitions."""

from __future__ import annotations

import pytest

from openai_sdk_helpers.structure import (
    BaseStructure,
    PromptStructure,
    StructureType,
    ToolSpec,
    build_tool_definitions,
)
from openai_sdk_helpers.structure.base import spec_field


class CustomStructure(BaseStructure):
    """A custom structure for testing."""

    query: str = spec_field("query", description="The search query.")
    limit: int = spec_field("limit", description="Maximum number of results.")


def test_tool_spec_creation():
    """Test creating a ToolSpec instance."""
    spec = ToolSpec(
        structure=PromptStructure,
        tool_name="test_tool",
        tool_description="A test tool description",
    )

    assert spec.structure == PromptStructure
    assert spec.tool_name == "test_tool"
    assert spec.tool_description == "A test tool description"


def test_tool_spec_frozen():
    """Test that ToolSpec is immutable."""
    spec = ToolSpec(
        structure=PromptStructure,
        tool_name="test_tool",
        tool_description="A test tool description",
    )

    with pytest.raises(Exception):  # dataclass frozen=True raises FrozenInstanceError
        spec.tool_name = "new_name"  # type: ignore


def test_build_tool_definitions_single_tool():
    """Test building a single tool definition."""
    specs = [
        ToolSpec(
            structure=CustomStructure,
            tool_name="search_tool",
            tool_description="Search for relevant information",
        )
    ]

    tools = build_tool_definitions(specs)

    assert isinstance(tools, list)
    assert len(tools) == 1

    tool = tools[0]
    assert tool["type"] == "function"
    assert tool["name"] == "search_tool"
    assert tool["description"] == "Search for relevant information"
    assert "parameters" in tool
    assert tool["strict"] is True


def test_build_tool_definitions_multiple_tools():
    """Test building multiple tool definitions."""
    specs = [
        ToolSpec(
            structure=PromptStructure,
            tool_name="web_agent",
            tool_description="Run a web research workflow for the provided prompt.",
        ),
        ToolSpec(
            structure=PromptStructure,
            tool_name="vector_agent",
            tool_description="Run a vector search workflow for the provided prompt.",
        ),
        ToolSpec(
            structure=CustomStructure,
            tool_name="custom_agent",
            tool_description="Run a custom workflow.",
        ),
    ]

    tools = build_tool_definitions(specs)

    assert isinstance(tools, list)
    assert len(tools) == 3

    # Check first tool
    assert tools[0]["name"] == "web_agent"
    assert (
        tools[0]["description"]
        == "Run a web research workflow for the provided prompt."
    )

    # Check second tool
    assert tools[1]["name"] == "vector_agent"
    assert (
        tools[1]["description"]
        == "Run a vector search workflow for the provided prompt."
    )

    # Check third tool
    assert tools[2]["name"] == "custom_agent"
    assert tools[2]["description"] == "Run a custom workflow."

    # Verify all tools have proper structure
    for tool in tools:
        assert tool["type"] == "function"
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["strict"] is True


def test_build_tool_definitions_empty_list():
    """Test building tool definitions from an empty list."""
    tools = build_tool_definitions([])

    assert isinstance(tools, list)
    assert len(tools) == 0


def test_structure_type_alias():
    """Test that StructureType is a valid type alias."""

    # This is more of a type checking test, but we can verify it's usable
    def check_structure(struct: StructureType) -> bool:
        return issubclass(struct, BaseStructure)

    assert check_structure(PromptStructure)
    assert check_structure(CustomStructure)


def test_tool_spec_with_different_structures():
    """Test ToolSpec works with various BaseStructure subclasses."""
    from openai_sdk_helpers.structure import SummaryStructure, ValidationResultStructure

    specs = [
        ToolSpec(
            structure=PromptStructure,
            tool_name="prompt_tool",
            tool_description="Process prompts",
        ),
        ToolSpec(
            structure=SummaryStructure,
            tool_name="summary_tool",
            tool_description="Generate summaries",
        ),
        ToolSpec(
            structure=ValidationResultStructure,
            tool_name="validation_tool",
            tool_description="Validate results",
        ),
    ]

    tools = build_tool_definitions(specs)

    assert len(tools) == 3
    assert tools[0]["name"] == "prompt_tool"
    assert tools[1]["name"] == "summary_tool"
    assert tools[2]["name"] == "validation_tool"


def test_tool_definitions_have_correct_schema():
    """Test that generated tool definitions contain valid schemas."""
    spec = ToolSpec(
        structure=CustomStructure,
        tool_name="test_tool",
        tool_description="Test tool",
    )

    tools = build_tool_definitions([spec])
    tool = tools[0]

    # Verify schema structure
    assert "parameters" in tool
    schema = tool["parameters"]
    assert "properties" in schema
    assert "query" in schema["properties"]
    assert "limit" in schema["properties"]


def test_build_tool_definitions_preserves_order():
    """Test that tool definitions are built in the same order as specs."""
    specs = [
        ToolSpec(structure=PromptStructure, tool_name="tool_a", tool_description="A"),
        ToolSpec(structure=PromptStructure, tool_name="tool_b", tool_description="B"),
        ToolSpec(structure=PromptStructure, tool_name="tool_c", tool_description="C"),
    ]

    tools = build_tool_definitions(specs)

    assert tools[0]["name"] == "tool_a"
    assert tools[1]["name"] == "tool_b"
    assert tools[2]["name"] == "tool_c"


def test_tool_spec_equality():
    """Test ToolSpec equality comparison."""
    spec1 = ToolSpec(
        structure=PromptStructure,
        tool_name="test",
        tool_description="Test tool",
    )
    spec2 = ToolSpec(
        structure=PromptStructure,
        tool_name="test",
        tool_description="Test tool",
    )
    spec3 = ToolSpec(
        structure=CustomStructure,
        tool_name="test",
        tool_description="Test tool",
    )

    assert spec1 == spec2
    assert spec1 != spec3
