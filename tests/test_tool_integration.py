"""Integration tests for tool wrapper unwrapping and async handler support."""

from __future__ import annotations

import asyncio
import json
from openai_sdk_helpers.tools import ToolSpec, tool_handler_factory
from openai_sdk_helpers.response.tool_call import parse_tool_arguments
from openai_sdk_helpers.structure import StructureBase
from openai_sdk_helpers.structure.base import spec_field


class MockToolCall:
    """Mock tool call object for testing."""

    def __init__(self, arguments: str, name: str = "test_tool"):
        self.arguments = arguments
        self.name = name


class TargetingInput(StructureBase):
    """Input model for targeting tool."""

    campaign_id: str = spec_field(
        "campaign_id", allow_null=False, description="Campaign identifier."
    )
    audience: str = spec_field(
        "audience", allow_null=False, description="Target audience."
    )
    budget: float = spec_field(
        "budget",
        allow_null=False,
        description="Campaign budget.",
        default=1000.0,
    )


class TargetingOutput(StructureBase):
    """Output model for targeting tool."""

    campaign_id: str = spec_field(
        "campaign_id", allow_null=False, description="Campaign identifier."
    )
    targeting_status: str = spec_field(
        "targeting_status", allow_null=False, description="Targeting status."
    )
    estimated_reach: int = spec_field(
        "estimated_reach", allow_null=False, description="Estimated reach."
    )


def test_unwrap_and_sync_handler_integration():
    """Test that wrapper unwrapping works with sync handler."""

    def propose_targeting(payload: TargetingInput) -> TargetingOutput:
        return TargetingOutput(
            campaign_id=payload.campaign_id,
            targeting_status="active",
            estimated_reach=50000,
        )

    handler = tool_handler_factory(
        propose_targeting,
        tool_spec=ToolSpec(
            input_structure=TargetingInput,
            output_structure=TargetingOutput,
            tool_name="propose_targeting",
            tool_description="Propose targeting for a campaign.",
        ),
    )

    # Arguments wrapped by tool name
    tool_call = MockToolCall(
        '{"propose_targeting": {"campaign_id": "c123", "audience": "tech", "budget": 5000}}',
        name="propose_targeting",
    )

    result = handler(tool_call)
    parsed = json.loads(result)

    assert parsed["campaign_id"] == "c123"
    assert parsed["targeting_status"] == "active"
    assert parsed["estimated_reach"] == 50000


def test_unwrap_and_async_handler_integration():
    """Test that wrapper unwrapping works with async handler."""

    async def async_propose_targeting(payload: TargetingInput) -> TargetingOutput:
        await asyncio.sleep(0.001)  # Simulate async work
        return TargetingOutput(
            campaign_id=payload.campaign_id,
            targeting_status="active",
            estimated_reach=75000,
        )

    handler = tool_handler_factory(
        async_propose_targeting,
        tool_spec=ToolSpec(
            input_structure=TargetingInput,
            output_structure=TargetingOutput,
            tool_name="async_propose_targeting",
            tool_description="Propose targeting asynchronously.",
        ),
    )

    # Arguments wrapped by snake_case tool name
    tool_call = MockToolCall(
        '{"async_propose_targeting": {"campaign_id": "c456", "audience": "finance", "budget": 10000}}',
        name="async_propose_targeting",
    )

    result = handler(tool_call)
    parsed = json.loads(result)

    assert parsed["campaign_id"] == "c456"
    assert parsed["targeting_status"] == "active"
    assert parsed["estimated_reach"] == 75000


def test_unwrap_pascal_case_with_async():
    """Test PascalCase wrapper unwrapping with async handler."""

    class AsyncToolInput(StructureBase):
        """Input structure for async tool."""

        value: int = spec_field("value", allow_null=False, description="Input value.")
        multiplier: int = spec_field(
            "multiplier",
            allow_null=False,
            description="Multiplier.",
            default=2,
        )

    class AsyncToolOutput(StructureBase):
        """Output structure for async tool."""

        result: int = spec_field("result", allow_null=False, description="Result.")

    async def async_tool(payload: AsyncToolInput) -> dict[str, int]:
        await asyncio.sleep(0.001)
        return {"result": payload.value * payload.multiplier}

    handler = tool_handler_factory(
        async_tool,
        tool_spec=ToolSpec(
            input_structure=AsyncToolInput,
            output_structure=AsyncToolOutput,
            tool_name="AsyncTool",
            tool_description="Multiply a value.",
        ),
    )

    # Wrapped with PascalCase key
    tool_call = MockToolCall(
        '{"AsyncTool": {"value": 10, "multiplier": 3}}', name="AsyncTool"
    )

    result = handler(tool_call)
    parsed = json.loads(result)

    assert parsed["result"] == 30


def test_flat_arguments_with_async_handler():
    """Test that flat (non-wrapped) arguments still work with async handlers."""

    class AsyncSearchInput(StructureBase):
        """Input structure for async search."""

        query: str = spec_field("query", allow_null=False, description="Query.")
        limit: int = spec_field(
            "limit",
            allow_null=False,
            description="Limit.",
            default=10,
        )

    class AsyncSearchOutput(StructureBase):
        """Output structure for async search."""

        query: str = spec_field("query", allow_null=False, description="Query.")
        limit: int = spec_field("limit", allow_null=False, description="Limit.")
        results: list[str] = spec_field(
            "results",
            allow_null=False,
            description="Results.",
        )

    async def async_search(
        payload: AsyncSearchInput,
    ) -> dict[str, int | str | list[str]]:
        await asyncio.sleep(0.001)
        return {
            "query": payload.query,
            "limit": payload.limit,
            "results": ["result1", "result2"],
        }

    handler = tool_handler_factory(
        async_search,
        tool_spec=ToolSpec(
            input_structure=AsyncSearchInput,
            output_structure=AsyncSearchOutput,
            tool_name="async_search",
            tool_description="Run an async search.",
        ),
    )

    # No wrapper - flat arguments
    tool_call = MockToolCall('{"query": "test", "limit": 5}', name="async_search")

    result = handler(tool_call)
    parsed = json.loads(result)

    assert parsed["query"] == "test"
    assert parsed["limit"] == 5
    assert len(parsed["results"]) == 2


def test_unwrap_with_async_in_event_loop():
    """Test wrapper unwrapping with async handler inside an event loop."""

    class AsyncCalculatorInput(StructureBase):
        """Input structure for async calculator."""

        operation: str = spec_field(
            "operation",
            allow_null=False,
            description="Operation name.",
        )
        a: int = spec_field("a", allow_null=False, description="First operand.")
        b: int = spec_field("b", allow_null=False, description="Second operand.")

    class AsyncCalculatorOutput(StructureBase):
        """Output structure for async calculator."""

        result: int = spec_field("result", allow_null=False, description="Result.")

    async def async_calculator(payload: AsyncCalculatorInput) -> dict[str, int]:
        await asyncio.sleep(0.001)
        if payload.operation == "add":
            return {"result": payload.a + payload.b}
        if payload.operation == "multiply":
            return {"result": payload.a * payload.b}
        return {"result": 0}

    handler = tool_handler_factory(
        async_calculator,
        tool_spec=ToolSpec(
            input_structure=AsyncCalculatorInput,
            output_structure=AsyncCalculatorOutput,
            tool_name="async_calculator",
            tool_description="Calculate async results.",
        ),
    )

    async def test_in_loop():
        # Wrapped arguments
        tool_call = MockToolCall(
            '{"async_calculator": {"operation": "multiply", "a": 7, "b": 8}}',
            name="async_calculator",
        )
        result = handler(tool_call)
        parsed = json.loads(result)
        assert parsed["result"] == 56

    # Run inside event loop
    asyncio.run(test_in_loop())


def test_parse_tool_arguments_directly():
    """Test parse_tool_arguments function handles all wrapper cases."""
    # Test parse_tool_arguments directly without going through handler factory

    # Wrapped by exact tool name
    result = parse_tool_arguments(
        '{"ExampleTool": {"key": "value"}}', tool_name="ExampleTool"
    )
    assert result == {"key": "value"}

    # Wrapped by snake_case
    result = parse_tool_arguments(
        '{"example_tool": {"key": "value"}}', tool_name="ExampleTool"
    )
    assert result == {"key": "value"}

    # Case insensitive
    result = parse_tool_arguments(
        '{"exampletool": {"key": "value"}}', tool_name="ExampleTool"
    )
    assert result == {"key": "value"}

    # No wrapper
    result = parse_tool_arguments('{"key": "value"}', tool_name="ExampleTool")
    assert result == {"key": "value"}
