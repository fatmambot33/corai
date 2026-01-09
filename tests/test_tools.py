"""Tests for tool handler utilities."""

from __future__ import annotations

import json
import pytest
from pydantic import BaseModel, ValidationError

from openai_sdk_helpers.tools import (
    serialize_tool_result,
    tool_handler_factory,
    StructureType,
    ToolSpec,
    build_tool_definitions,
)
from openai_sdk_helpers.response.tool_call import parse_tool_arguments
from openai_sdk_helpers.structure import (
    BaseStructure,
    PromptStructure,
    SummaryStructure,
    ValidationResultStructure,
)
from openai_sdk_helpers.structure.base import spec_field


class SampleInput(BaseModel):
    """Sample Pydantic model for testing."""

    query: str
    limit: int = 10


class SampleOutput(BaseModel):
    """Sample output model."""

    results: list[str]
    count: int


def test_serialize_tool_result_with_pydantic():
    """Test serialization of Pydantic models."""
    output = SampleOutput(results=["result1", "result2"], count=2)
    serialized = serialize_tool_result(output)

    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed["results"] == ["result1", "result2"]
    assert parsed["count"] == 2


def test_serialize_tool_result_with_list():
    """Test serialization of lists."""
    result = ["item1", "item2", "item3"]
    serialized = serialize_tool_result(result)

    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed == result


def test_serialize_tool_result_with_dict():
    """Test serialization of dictionaries."""
    result = {"key": "value", "number": 42}
    serialized = serialize_tool_result(result)

    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed == result


def test_serialize_tool_result_with_string():
    """Test serialization of plain strings."""
    result = "plain text result"
    serialized = serialize_tool_result(result)

    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert parsed == result


def test_serialize_tool_result_with_primitives():
    """Test serialization of primitive types."""
    assert serialize_tool_result(42) == "42"
    assert serialize_tool_result(3.14) == "3.14"
    assert serialize_tool_result(True) == "true"
    assert serialize_tool_result(None) == "null"


def test_parse_tool_arguments_with_tool_name():
    """Test enhanced parse_tool_arguments with tool name."""
    args = '{"key": "value"}'
    result = parse_tool_arguments(args, tool_name="test_tool")
    assert result == {"key": "value"}


def test_parse_tool_arguments_error_includes_tool_name():
    """Test that parse errors include tool name for context."""
    invalid_args = '{"key": invalid}'

    with pytest.raises(ValueError) as exc_info:
        parse_tool_arguments(invalid_args, tool_name="my_tool")

    error_msg = str(exc_info.value)
    assert "my_tool" in error_msg
    assert "Raw payload" in error_msg


def test_parse_tool_arguments_truncates_long_payload():
    """Test that long payloads are truncated in error messages."""
    # Create an invalid payload longer than 100 characters
    long_payload = '{"key": invalid_value_' + "x" * 200 + "}"

    with pytest.raises(ValueError) as exc_info:
        parse_tool_arguments(long_payload, tool_name="test")

    error_msg = str(exc_info.value)
    assert "..." in error_msg  # Should be truncated


class MockToolCall:
    """Mock tool call object for testing."""

    def __init__(self, arguments: str, name: str = "test_tool"):
        self.arguments = arguments
        self.name = name


def test_tool_handler_factory_basic():
    """Test basic tool handler without validation."""

    def simple_tool(query: str, limit: int = 10):
        return {"query": query, "limit": limit}

    handler = tool_handler_factory(simple_tool)

    tool_call = MockToolCall('{"query": "test", "limit": 5}')
    result = handler(tool_call)

    # Result should be JSON string
    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert parsed["limit"] == 5


def test_tool_handler_factory_with_validation():
    """Test tool handler with Pydantic validation."""

    def search_tool(query: str, limit: int = 10):
        return SampleOutput(results=[f"result for {query}"], count=1)

    handler = tool_handler_factory(search_tool, input_model=SampleInput)

    tool_call = MockToolCall('{"query": "test search", "limit": 20}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["results"] == ["result for test search"]
    assert parsed["count"] == 1


def test_tool_handler_factory_validation_failure():
    """Test that validation errors are raised with context."""

    def dummy_tool(query: str, limit: int):
        return {}

    handler = tool_handler_factory(dummy_tool, input_model=SampleInput)

    # Missing required field 'query'
    tool_call = MockToolCall('{"limit": 10}', name="search")

    with pytest.raises(ValidationError):
        handler(tool_call)


def test_tool_handler_factory_with_defaults():
    """Test that default values work correctly."""

    def tool_with_defaults(query: str, limit: int = 10, offset: int = 0):
        return {"query": query, "limit": limit, "offset": offset}

    handler = tool_handler_factory(tool_with_defaults)

    # Only provide query, use defaults
    tool_call = MockToolCall('{"query": "test"}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert parsed["limit"] == 10
    assert parsed["offset"] == 0


def test_tool_handler_factory_argument_parsing_error():
    """Test that argument parsing errors include tool name."""

    def simple_tool(query: str):
        return {"query": query}

    handler = tool_handler_factory(simple_tool)

    # Invalid JSON
    tool_call = MockToolCall("invalid json", name="my_tool")

    with pytest.raises(ValueError) as exc_info:
        handler(tool_call)

    error_msg = str(exc_info.value)
    assert "my_tool" in error_msg


def test_tool_handler_factory_returns_string():
    """Test that the result from handler is a JSON string."""

    def tool_returning_list():
        return ["a", "b", "c"]

    handler = tool_handler_factory(tool_returning_list)

    tool_call = MockToolCall("{}")
    result = handler(tool_call)

    # Should be a string
    assert isinstance(result, str)

    # Should be valid JSON
    parsed = json.loads(result)
    assert parsed == ["a", "b", "c"]


# ToolSpec and build_tool_definitions tests
# ==========================================


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


def test_tool_spec_with_output_structure():
    """Test ToolSpec with separate input and output structures."""
    spec = ToolSpec(
        structure=PromptStructure,
        tool_name="summarizer",
        tool_description="Summarize the provided prompt",
        output_structure=SummaryStructure,
    )

    assert spec.structure == PromptStructure
    assert spec.output_structure == SummaryStructure
    assert spec.tool_name == "summarizer"

    # The tool definition should still use the input structure
    tools = build_tool_definitions([spec])
    tool = tools[0]

    # Verify it uses the input structure (PromptStructure) for parameters
    assert tool["name"] == "summarizer"
    schema = tool["parameters"]
    assert "properties" in schema
    # PromptStructure has "prompt" field
    assert "prompt" in schema["properties"]
    # SummaryStructure fields should NOT be in the tool parameters
    assert "topics" not in schema["properties"]


def test_tool_spec_output_structure_optional():
    """Test that output_structure is optional and defaults to None."""
    spec = ToolSpec(
        structure=PromptStructure,
        tool_name="test_tool",
        tool_description="Test tool",
    )

    assert spec.output_structure is None


def test_tool_spec_with_different_io_structures():
    """Test multiple tools with different input/output combinations."""
    specs = [
        # Tool with same input/output (implicit)
        ToolSpec(
            structure=PromptStructure,
            tool_name="echo",
            tool_description="Echo the prompt",
        ),
        # Tool with explicit different output
        ToolSpec(
            structure=PromptStructure,
            tool_name="summarize",
            tool_description="Summarize the prompt",
            output_structure=SummaryStructure,
        ),
        # Tool with explicit different output
        ToolSpec(
            structure=PromptStructure,
            tool_name="validate",
            tool_description="Validate the prompt",
            output_structure=ValidationResultStructure,
        ),
    ]

    tools = build_tool_definitions(specs)

    assert len(tools) == 3
    # All should use PromptStructure as input (have "prompt" field)
    for tool in tools:
        assert "prompt" in tool["parameters"]["properties"]


# Tests for wrapper unwrapping functionality
# ===========================================


def test_parse_tool_arguments_unwraps_matching_wrapper():
    """Test that parse_tool_arguments unwraps arguments wrapped by class name."""
    # Wrapper matches tool name exactly
    args = '{"ExampleTool": {"key": "value", "number": 42}}'
    result = parse_tool_arguments(args, tool_name="ExampleTool")
    assert result == {"key": "value", "number": 42}


def test_parse_tool_arguments_unwraps_snake_case_wrapper():
    """Test unwrapping with snake_case wrapper key."""
    # Wrapper is snake_case version of tool name
    args = '{"example_tool": {"key": "value"}}'
    result = parse_tool_arguments(args, tool_name="ExampleTool")
    assert result == {"key": "value"}


def test_parse_tool_arguments_unwraps_case_insensitive():
    """Test case-insensitive wrapper unwrapping."""
    # Wrapper key differs only in case
    args = '{"exampletool": {"key": "value"}}'
    result = parse_tool_arguments(args, tool_name="ExampleTool")
    assert result == {"key": "value"}


def test_parse_tool_arguments_no_unwrap_multiple_keys():
    """Test that multi-key dicts are not unwrapped."""
    # Multiple keys - should not unwrap even if one matches
    args = '{"ExampleTool": {"key": "value"}, "other": "data"}'
    result = parse_tool_arguments(args, tool_name="ExampleTool")
    assert result == {"ExampleTool": {"key": "value"}, "other": "data"}


def test_parse_tool_arguments_no_unwrap_non_dict_value():
    """Test that wrappers with non-dict values are not unwrapped."""
    # Value is not a dict
    args = '{"ExampleTool": "string_value"}'
    result = parse_tool_arguments(args, tool_name="ExampleTool")
    assert result == {"ExampleTool": "string_value"}


def test_parse_tool_arguments_no_unwrap_non_matching():
    """Test that non-matching wrappers are not unwrapped."""
    # Wrapper doesn't match tool name
    args = '{"DifferentTool": {"key": "value"}}'
    result = parse_tool_arguments(args, tool_name="ExampleTool")
    assert result == {"DifferentTool": {"key": "value"}}


def test_parse_tool_arguments_flat_dict_unchanged():
    """Test that flat dicts without wrappers work as before."""
    args = '{"key": "value", "number": 42}'
    result = parse_tool_arguments(args, tool_name="ExampleTool")
    assert result == {"key": "value", "number": 42}


# Tests for async tool handler support
# =====================================


def test_tool_handler_factory_with_async_function():
    """Test tool handler factory with async function."""
    import asyncio

    async def async_search(query: str, limit: int = 10):
        await asyncio.sleep(0.001)  # Simulate async work
        return {"query": query, "limit": limit, "async": True}

    handler = tool_handler_factory(async_search)

    tool_call = MockToolCall('{"query": "test", "limit": 5}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert parsed["limit"] == 5
    assert parsed["async"] is True


def test_tool_handler_factory_async_with_validation():
    """Test async tool handler with Pydantic validation."""
    import asyncio

    async def async_search(query: str, limit: int = 10):
        await asyncio.sleep(0.001)
        return SampleOutput(results=[f"async result for {query}"], count=1)

    handler = tool_handler_factory(async_search, input_model=SampleInput)

    tool_call = MockToolCall('{"query": "async test", "limit": 20}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["results"] == ["async result for async test"]
    assert parsed["count"] == 1


def test_tool_handler_factory_async_raises_validation_error():
    """Test that async handlers properly raise validation errors."""
    import asyncio

    async def async_tool(query: str, limit: int):
        await asyncio.sleep(0.001)
        return {}

    handler = tool_handler_factory(async_tool, input_model=SampleInput)

    # Missing required field 'query'
    tool_call = MockToolCall('{"limit": 10}', name="async_search")

    with pytest.raises(ValidationError):
        handler(tool_call)


def test_tool_handler_factory_async_inside_event_loop():
    """Test async handler when already inside an event loop."""
    import asyncio

    async def async_tool(value: int):
        await asyncio.sleep(0.001)
        return {"result": value * 2}

    handler = tool_handler_factory(async_tool)

    async def test_in_loop():
        tool_call = MockToolCall('{"value": 21}')
        result = handler(tool_call)
        parsed = json.loads(result)
        assert parsed["result"] == 42

    # Run the test inside an event loop
    asyncio.run(test_in_loop())


def test_tool_handler_factory_async_complex_return():
    """Test async handler with complex return types."""
    import asyncio

    async def async_complex_tool(items: list[str]):
        await asyncio.sleep(0.001)
        return {
            "processed": [item.upper() for item in items],
            "count": len(items),
            "nested": {"status": "success"},
        }

    handler = tool_handler_factory(async_complex_tool)

    tool_call = MockToolCall('{"items": ["a", "b", "c"]}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["processed"] == ["A", "B", "C"]
    assert parsed["count"] == 3
    assert parsed["nested"]["status"] == "success"
