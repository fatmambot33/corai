def test_dummy_for_coverage(openai_settings):
    """Dummy test to increase coverage by exercising __repr__."""
    from openai_sdk_helpers.response.base import ResponseBase

    class DummyStruct:
        pass

    r = ResponseBase(
        name="test",
        instructions="hi",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    assert "ResponseBase" in repr(r)
