"""Tests for tool handler utilities."""

from __future__ import annotations

import asyncio
import json
import pytest
from pydantic import ValidationError

from openai_sdk_helpers.tools import (
    serialize_tool_result,
    tool_handler_factory,
    unserialize_tool_arguments,
    StructureType,
    ToolSpec,
    build_tool_definitions,
    parse_tool_arguments,
)

from openai_sdk_helpers.structure import (
    StructureBase,
    PromptStructure,
    SummaryStructure,
    ValidationResultStructure,
)
from openai_sdk_helpers.structure.base import spec_field




class MockToolCall:
    """Mock tool call object for testing."""

    def __init__(self, arguments: str, name: str = "test_tool"):
        self.arguments = arguments
        self.name = name


def test_tool_handler_factory_basic():
    """Test basic tool handler without validation."""

    def simple_tool(payload: BasicInputStructure) -> dict[str, int | str]:
        return payload.model_dump()

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=BasicOutputStructure,
        tool_name="simple_tool",
        tool_description="Return the query and limits.",
    )
    handler = tool_handler_factory(simple_tool, tool_spec=spec)

    tool_call = MockToolCall('{"query": "test", "limit": 5}')
    result = handler(tool_call)

    # Result should be JSON string
    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert parsed["limit"] == 5


def test_tool_handler_factory_with_validation():
    """Test tool handler with Pydantic validation."""

    def search_tool(payload: BasicInputStructure) -> dict[str, int | str]:
        return payload.model_dump()

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=BasicOutputStructure,
        tool_name="search",
        tool_description="Run a search query.",
    )
    handler = tool_handler_factory(search_tool, tool_spec=spec)

    tool_call = MockToolCall('{"query": "test search", "limit": 20}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["query"] == "test search"
    assert parsed["limit"] == 20


def test_tool_handler_factory_validation_failure():
    """Test that validation errors are raised with context."""

    def dummy_tool(payload: BasicInputStructure) -> dict[str, int | str]:
        return payload.model_dump()

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=BasicOutputStructure,
        tool_name="search",
        tool_description="Run a search query.",
    )
    handler = tool_handler_factory(dummy_tool, tool_spec=spec)

    # Missing required field 'query'
    tool_call = MockToolCall('{"limit": 10}', name="search")

    with pytest.raises(ValidationError):
        handler(tool_call)


def test_tool_handler_factory_with_defaults():
    """Test that default values work correctly."""

    def tool_with_defaults(payload: BasicInputStructure) -> dict[str, int | str]:
        return payload.model_dump()

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=BasicOutputStructure,
        tool_name="defaults",
        tool_description="Return defaults for the query.",
    )
    handler = tool_handler_factory(tool_with_defaults, tool_spec=spec)

    # Only provide query, use defaults
    tool_call = MockToolCall('{"query": "test"}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert parsed["limit"] == 10
    assert parsed["offset"] == 0


def test_tool_handler_factory_argument_parsing_error():
    """Test that argument parsing errors include tool name."""

    def simple_tool(payload: BasicInputStructure) -> dict[str, int | str]:
        return payload.model_dump()

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=BasicOutputStructure,
        tool_name="simple_tool",
        tool_description="Return the query and limits.",
    )
    handler = tool_handler_factory(simple_tool, tool_spec=spec)

    # Invalid JSON
    tool_call = MockToolCall("invalid json", name="my_tool")

    with pytest.raises(ValueError) as exc_info:
        handler(tool_call)

    error_msg = str(exc_info.value)
    assert "my_tool" in error_msg


def test_tool_handler_factory_returns_string():
    """Test that the result from handler is a JSON string."""

    def tool_returning_list(_: BasicInputStructure) -> dict[str, list[str]]:
        return {"items": ["a", "b", "c"]}

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=ItemsOutputStructure,
        tool_name="list_tool",
        tool_description="Return a list of items.",
    )
    handler = tool_handler_factory(tool_returning_list, tool_spec=spec)

    tool_call = MockToolCall('{"query": "test"}')
    result = handler(tool_call)

    # Should be a string
    assert isinstance(result, str)

    # Should be valid JSON
    parsed = json.loads(result)
    assert parsed == {"items": ["a", "b", "c"]}


def test_serialize_tool_result_requires_output_structure():
    """Test that serialize_tool_result requires an output structure."""
    spec = ToolSpec(
        input_structure=BasicInputStructure,
        tool_name="missing_output",
        tool_description="Missing output structure.",
    )

    with pytest.raises(ValueError):
        serialize_tool_result({"query": "test"}, tool_spec=spec)


def test_unserialize_tool_arguments_returns_structure():
    """Test unserialize_tool_arguments returns a validated structure."""
    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=BasicOutputStructure,
        tool_name="unserialize",
        tool_description="Unserialize arguments.",
    )
    tool_call = MockToolCall('{"query": "test", "limit": 3, "offset": 1}')

    result = unserialize_tool_arguments(tool_call, tool_spec=spec)

    assert isinstance(result, BasicInputStructure)
    assert result.query == "test"
    assert result.limit == 3
    assert result.offset == 1


# ToolSpec and build_tool_definitions tests
# ==========================================


class CustomStructure(StructureBase):
    """A custom structure for testing."""

    query: str = spec_field("query", description="The search query.")
    limit: int = spec_field("limit", description="Maximum number of results.")


class BasicInputStructure(StructureBase):
    """Input structure for tool handler tests."""

    query: str = spec_field("query", allow_null=False, description="Search query.")
    limit: int = spec_field(
        "limit",
        allow_null=False,
        description="Maximum number of results.",
        default=10,
    )
    offset: int = spec_field(
        "offset",
        allow_null=False,
        description="Pagination offset.",
        default=0,
    )


class BasicOutputStructure(StructureBase):
    """Output structure for tool handler tests."""

    query: str = spec_field("query", allow_null=False, description="Search query.")
    limit: int = spec_field(
        "limit",
        allow_null=False,
        description="Maximum number of results.",
    )
    offset: int = spec_field(
        "offset",
        allow_null=False,
        description="Pagination offset.",
    )


class ItemsOutputStructure(StructureBase):
    """Output structure for list results."""

    items: list[str] = spec_field(
        "items",
        allow_null=False,
        description="Returned items.",
    )


class AsyncQueryOutputStructure(StructureBase):
    """Output structure for async query results."""

    query: str = spec_field("query", allow_null=False, description="Search query.")
    limit: int = spec_field(
        "limit",
        allow_null=False,
        description="Maximum number of results.",
    )
    async_flag: bool = spec_field(
        "async_flag",
        allow_null=False,
        description="Whether the call ran asynchronously.",
    )


class AsyncValueInputStructure(StructureBase):
    """Input structure for async value tests."""

    value: int = spec_field(
        "value",
        allow_null=False,
        description="Input value.",
    )


class AsyncValueOutputStructure(StructureBase):
    """Output structure for async value tests."""

    result: int = spec_field(
        "result",
        allow_null=False,
        description="Doubled result.",
    )


class AsyncItemsInputStructure(StructureBase):
    """Input structure for async list processing tests."""

    items: list[str] = spec_field(
        "items",
        allow_null=False,
        description="Input items.",
    )


class AsyncComplexOutputStructure(StructureBase):
    """Output structure for async list processing tests."""

    processed: list[str] = spec_field(
        "processed",
        allow_null=False,
        description="Processed items.",
    )
    count: int = spec_field(
        "count",
        allow_null=False,
        description="Number of items.",
    )
    nested: dict[str, str] = spec_field(
        "nested",
        allow_null=False,
        description="Nested status payload.",
    )


def test_tool_spec_creation():
    """Test creating a ToolSpec instance."""
    spec = ToolSpec(
        input_structure=PromptStructure,
        tool_name="test_tool",
        tool_description="A test tool description",
    )

    assert spec.input_structure == PromptStructure
    assert spec.tool_name == "test_tool"
    assert spec.tool_description == "A test tool description"


def test_tool_spec_frozen():
    """Test that ToolSpec is immutable."""
    spec = ToolSpec(
        input_structure=PromptStructure,
        tool_name="test_tool",
        tool_description="A test tool description",
    )

    with pytest.raises(Exception):  # dataclass frozen=True raises FrozenInstanceError
        spec.tool_name = "new_name"  # type: ignore


def test_build_tool_definitions_single_tool():
    """Test building a single tool definition."""
    specs = [
        ToolSpec(
            input_structure=CustomStructure,
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
            input_structure=PromptStructure,
            tool_name="web_agent",
            tool_description="Run a web research workflow for the provided prompt.",
        ),
        ToolSpec(
            input_structure=PromptStructure,
            tool_name="vector_agent",
            tool_description="Run a vector search workflow for the provided prompt.",
        ),
        ToolSpec(
            input_structure=CustomStructure,
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
        return issubclass(struct, StructureBase)

    assert check_structure(PromptStructure)
    assert check_structure(CustomStructure)


def test_tool_spec_with_different_structures():
    """Test ToolSpec works with various StructureBase subclasses."""
    specs = [
        ToolSpec(
            input_structure=PromptStructure,
            tool_name="prompt_tool",
            tool_description="Process prompts",
        ),
        ToolSpec(
            input_structure=SummaryStructure,
            tool_name="summary_tool",
            tool_description="Generate summaries",
        ),
        ToolSpec(
            input_structure=ValidationResultStructure,
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
        input_structure=CustomStructure,
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
        ToolSpec(
            input_structure=PromptStructure, tool_name="tool_a", tool_description="A"
        ),
        ToolSpec(
            input_structure=PromptStructure, tool_name="tool_b", tool_description="B"
        ),
        ToolSpec(
            input_structure=PromptStructure, tool_name="tool_c", tool_description="C"
        ),
    ]

    tools = build_tool_definitions(specs)

    assert tools[0]["name"] == "tool_a"
    assert tools[1]["name"] == "tool_b"
    assert tools[2]["name"] == "tool_c"


def test_tool_spec_equality():
    """Test ToolSpec equality comparison."""
    spec1 = ToolSpec(
        input_structure=PromptStructure,
        tool_name="test",
        tool_description="Test tool",
    )
    spec2 = ToolSpec(
        input_structure=PromptStructure,
        tool_name="test",
        tool_description="Test tool",
    )
    spec3 = ToolSpec(
        input_structure=CustomStructure,
        tool_name="test",
        tool_description="Test tool",
    )

    assert spec1 == spec2
    assert spec1 != spec3


def test_tool_spec_with_output_structure():
    """Test ToolSpec with separate input and output structures."""
    spec = ToolSpec(
        input_structure=PromptStructure,
        tool_name="summarizer",
        tool_description="Summarize the provided prompt",
        output_structure=SummaryStructure,
    )

    assert spec.input_structure == PromptStructure
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
        input_structure=PromptStructure,
        tool_name="test_tool",
        tool_description="Test tool",
    )

    assert spec.output_structure is None


def test_tool_spec_with_different_io_structures():
    """Test multiple tools with different input/output combinations."""
    specs = [
        # Tool with same input/output (implicit)
        ToolSpec(
            input_structure=PromptStructure,
            tool_name="echo",
            tool_description="Echo the prompt",
        ),
        # Tool with explicit different output
        ToolSpec(
            input_structure=PromptStructure,
            tool_name="summarize",
            tool_description="Summarize the prompt",
            output_structure=SummaryStructure,
        ),
        # Tool with explicit different output
        ToolSpec(
            input_structure=PromptStructure,
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

    async def async_search(payload: BasicInputStructure) -> dict[str, int | str | bool]:
        await asyncio.sleep(0.001)  # Simulate async work
        return {"query": payload.query, "limit": payload.limit, "async_flag": True}

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=AsyncQueryOutputStructure,
        tool_name="async_search",
        tool_description="Run an async search query.",
    )
    handler = tool_handler_factory(async_search, tool_spec=spec)

    tool_call = MockToolCall('{"query": "test", "limit": 5}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert parsed["limit"] == 5
    assert parsed["async_flag"] is True


def test_tool_handler_factory_async_with_validation():
    """Test async tool handler with Pydantic validation."""

    async def async_search(payload: BasicInputStructure) -> dict[str, int | str]:
        await asyncio.sleep(0.001)
        return {
            "query": payload.query,
            "limit": payload.limit,
            "offset": payload.offset,
        }

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=BasicOutputStructure,
        tool_name="async_search",
        tool_description="Run an async search query.",
    )
    handler = tool_handler_factory(async_search, tool_spec=spec)

    tool_call = MockToolCall('{"query": "async test", "limit": 20}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["query"] == "async test"
    assert parsed["limit"] == 20
    assert parsed["offset"] == 0


def test_tool_handler_factory_async_raises_validation_error():
    """Test that async handlers properly raise validation errors."""

    async def async_tool(payload: BasicInputStructure) -> dict[str, int | str]:
        await asyncio.sleep(0.001)
        return payload.model_dump()

    spec = ToolSpec(
        input_structure=BasicInputStructure,
        output_structure=BasicOutputStructure,
        tool_name="async_search",
        tool_description="Run an async search query.",
    )
    handler = tool_handler_factory(async_tool, tool_spec=spec)

    # Missing required field 'query'
    tool_call = MockToolCall('{"limit": 10}', name="async_search")

    with pytest.raises(ValidationError):
        handler(tool_call)


def test_tool_handler_factory_async_inside_event_loop():
    """Test async handler when already inside an event loop."""

    async def async_tool(payload: AsyncValueInputStructure) -> dict[str, int]:
        await asyncio.sleep(0.001)
        return {"result": payload.value * 2}

    spec = ToolSpec(
        input_structure=AsyncValueInputStructure,
        output_structure=AsyncValueOutputStructure,
        tool_name="async_value",
        tool_description="Double the input value.",
    )
    handler = tool_handler_factory(async_tool, tool_spec=spec)

    async def test_in_loop():
        tool_call = MockToolCall('{"value": 21}')
        result = handler(tool_call)
        parsed = json.loads(result)
        assert parsed["result"] == 42

    # Run the test inside an event loop
    asyncio.run(test_in_loop())


def test_tool_handler_factory_async_complex_return():
    """Test async handler with complex return types."""

    async def async_complex_tool(
        payload: AsyncItemsInputStructure,
    ) -> dict[str, list[str] | int | dict[str, str]]:
        await asyncio.sleep(0.001)
        return {
            "processed": [item.upper() for item in payload.items],
            "count": len(payload.items),
            "nested": {"status": "success"},
        }

    spec = ToolSpec(
        input_structure=AsyncItemsInputStructure,
        output_structure=AsyncComplexOutputStructure,
        tool_name="async_complex",
        tool_description="Process items asynchronously.",
    )
    handler = tool_handler_factory(async_complex_tool, tool_spec=spec)

    tool_call = MockToolCall('{"items": ["a", "b", "c"]}')
    result = handler(tool_call)

    parsed = json.loads(result)
    assert parsed["processed"] == ["A", "B", "C"]
    assert parsed["count"] == 3
    assert parsed["nested"]["status"] == "success"
