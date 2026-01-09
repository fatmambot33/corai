"""Integration tests for tool wrapper unwrapping and async handler support."""

from __future__ import annotations

import asyncio
import json
import pytest
from pydantic import BaseModel

from openai_sdk_helpers.tools import tool_handler_factory
from openai_sdk_helpers.response.tool_call import parse_tool_arguments


class MockToolCall:
    """Mock tool call object for testing."""

    def __init__(self, arguments: str, name: str = "test_tool"):
        self.arguments = arguments
        self.name = name


class TargetingInput(BaseModel):
    """Input model for targeting tool."""

    campaign_id: str
    audience: str
    budget: float = 1000.0


class TargetingOutput(BaseModel):
    """Output model for targeting tool."""

    campaign_id: str
    targeting_status: str
    estimated_reach: int


def test_unwrap_and_sync_handler_integration():
    """Test that wrapper unwrapping works with sync handler."""

    def propose_targeting(campaign_id: str, audience: str, budget: float):
        return TargetingOutput(
            campaign_id=campaign_id,
            targeting_status="active",
            estimated_reach=50000,
        )

    handler = tool_handler_factory(propose_targeting, input_model=TargetingInput)

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

    async def async_propose_targeting(campaign_id: str, audience: str, budget: float):
        await asyncio.sleep(0.001)  # Simulate async work
        return TargetingOutput(
            campaign_id=campaign_id,
            targeting_status="active",
            estimated_reach=75000,
        )

    handler = tool_handler_factory(
        async_propose_targeting, input_model=TargetingInput
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

    async def async_tool(value: int, multiplier: int = 2):
        await asyncio.sleep(0.001)
        return {"result": value * multiplier}

    handler = tool_handler_factory(async_tool)

    # Wrapped with PascalCase key
    tool_call = MockToolCall(
        '{"AsyncTool": {"value": 10, "multiplier": 3}}', name="AsyncTool"
    )

    result = handler(tool_call)
    parsed = json.loads(result)

    assert parsed["result"] == 30


def test_flat_arguments_with_async_handler():
    """Test that flat (non-wrapped) arguments still work with async handlers."""

    async def async_search(query: str, limit: int = 10):
        await asyncio.sleep(0.001)
        return {"query": query, "limit": limit, "results": ["result1", "result2"]}

    handler = tool_handler_factory(async_search)

    # No wrapper - flat arguments
    tool_call = MockToolCall('{"query": "test", "limit": 5}', name="async_search")

    result = handler(tool_call)
    parsed = json.loads(result)

    assert parsed["query"] == "test"
    assert parsed["limit"] == 5
    assert len(parsed["results"]) == 2


def test_unwrap_with_async_in_event_loop():
    """Test wrapper unwrapping with async handler inside an event loop."""

    async def async_calculator(operation: str, a: int, b: int):
        await asyncio.sleep(0.001)
        if operation == "add":
            return {"result": a + b}
        elif operation == "multiply":
            return {"result": a * b}
        return {"result": 0}

    handler = tool_handler_factory(async_calculator)

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
    # Test direct usage without handler factory

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
