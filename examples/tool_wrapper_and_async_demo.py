"""Example demonstrating tool wrapper unwrapping and async handler support.

This example shows how the new features solve the two problems outlined:
1. Tool schema wrapper handling - automatic unwrapping of wrapped payloads
2. Async tool handler support - built-in async function support with event loop detection
"""

import asyncio
from pydantic import BaseModel
from openai_sdk_helpers.tools import tool_handler_factory


# Define input/output models
class TargetingInput(BaseModel):
    """Input parameters for the targeting tool."""

    campaign_id: str
    audience: str
    budget: float = 1000.0


class TargetingOutput(BaseModel):
    """Output from the targeting tool."""

    campaign_id: str
    targeting_status: str
    estimated_reach: int
    message: str


# Example 1: Async tool with automatic wrapper unwrapping
# ========================================================
async def propose_targeting(
    campaign_id: str, audience: str, budget: float
) -> TargetingOutput:
    """Async tool that proposes targeting parameters for a campaign.

    Before: Required custom _run_awaitable/thread handling to work in event loops.
    After: tool_handler_factory automatically handles async functions with event loop detection.
    """
    # Simulate async API call
    await asyncio.sleep(0.01)

    # Calculate estimated reach based on budget
    estimated_reach = int(budget * 10)

    return TargetingOutput(
        campaign_id=campaign_id,
        targeting_status="active",
        estimated_reach=estimated_reach,
        message=f"Targeting configured for {audience} audience",
    )


# Create handler - no special async handling needed!
# Before: Had to implement custom thread/event loop logic
# After: Just pass async function directly
async_handler = tool_handler_factory(propose_targeting, input_model=TargetingInput)


# Example 2: Handler automatically unwraps payloads
# ==================================================
class MockToolCall:
    """Mock tool call for demonstration."""

    def __init__(self, arguments: str, name: str):
        self.arguments = arguments
        self.name = name


def demonstrate_wrapper_unwrapping():
    """Show how wrapper unwrapping works automatically."""
    print("=" * 70)
    print("Example 1: Wrapper Unwrapping")
    print("=" * 70)

    # Scenario 1: Response wraps arguments under tool name
    # Before: Would fail validation with strict=True, additionalProperties=False
    # After: Automatically unwrapped before validation
    wrapped_call = MockToolCall(
        arguments='{"propose_targeting": {"campaign_id": "c123", "audience": "tech", "budget": 5000}}',
        name="propose_targeting",
    )

    result = async_handler(wrapped_call)
    print("\n✓ Wrapped payload (tool name key):")
    print(f"  Input:  {wrapped_call.arguments}")
    print(f"  Output: {result}")

    # Scenario 2: Response uses snake_case wrapper
    snake_case_call = MockToolCall(
        arguments='{"propose_targeting": {"campaign_id": "c456", "audience": "finance", "budget": 10000}}',
        name="propose_targeting",
    )

    result = async_handler(snake_case_call)
    print("\n✓ Wrapped payload (snake_case key):")
    print(f"  Input:  {snake_case_call.arguments}")
    print(f"  Output: {result}")

    # Scenario 3: Flat arguments (no wrapper) still work
    flat_call = MockToolCall(
        arguments='{"campaign_id": "c789", "audience": "healthcare", "budget": 15000}',
        name="propose_targeting",
    )

    result = async_handler(flat_call)
    print("\n✓ Flat payload (no wrapper):")
    print(f"  Input:  {flat_call.arguments}")
    print(f"  Output: {result}")


async def demonstrate_async_support():
    """Show how async tools work in various contexts."""
    print("\n" + "=" * 70)
    print("Example 2: Async Tool Handler Support")
    print("=" * 70)

    # Scenario 1: Async tool called outside event loop
    print("\n✓ Async handler called from sync context:")
    tool_call = MockToolCall(
        arguments='{"campaign_id": "async1", "audience": "education", "budget": 2000}',
        name="propose_targeting",
    )
    result = async_handler(tool_call)
    print(f"  Result: {result[:80]}...")

    # Scenario 2: Async tool called inside event loop
    print("\n✓ Async handler called from async context (we're in one now):")
    tool_call2 = MockToolCall(
        arguments='{"campaign_id": "async2", "audience": "retail", "budget": 8000}',
        name="propose_targeting",
    )
    result2 = async_handler(tool_call2)
    print(f"  Result: {result2[:80]}...")

    # Scenario 3: Multiple async calls work correctly
    print("\n✓ Multiple async calls in sequence:")
    for i in range(3):
        tool_call_seq = MockToolCall(
            arguments=f'{{"campaign_id": "seq{i}", "audience": "audience{i}", "budget": {(i+1)*1000}}}',
            name="propose_targeting",
        )
        result_seq = async_handler(tool_call_seq)
        print(f"  Call {i+1}: Success")


# Example 3: Sync tool still works as before
# ===========================================
def sync_tool(name: str, value: int) -> dict:
    """Synchronize tool for comparison."""
    return {"name": name, "doubled": value * 2}


sync_handler = tool_handler_factory(sync_tool)


def demonstrate_backward_compatibility():
    """Show that sync tools still work as before."""
    print("\n" + "=" * 70)
    print("Example 3: Backward Compatibility")
    print("=" * 70)

    tool_call = MockToolCall(
        arguments='{"name": "test", "value": 21}',
        name="sync_tool",
    )
    result = sync_handler(tool_call)
    print(f"\n✓ Sync handler still works:")
    print(f"  Input:  {tool_call.arguments}")
    print(f"  Output: {result}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_wrapper_unwrapping()

    # Run async example (creates event loop)
    asyncio.run(demonstrate_async_support())

    demonstrate_backward_compatibility()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        """
✓ Wrapper unwrapping: Arguments wrapped by tool name are automatically unwrapped
✓ Async support: Async functions work seamlessly with event loop detection
✓ Backward compatible: Sync tools continue to work as before
✓ No manual thread handling: Event loop management is automatic
✓ Validation still works: Pydantic validation happens after unwrapping
    """
    )
