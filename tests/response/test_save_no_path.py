from openai_sdk_helpers.response.base import BaseResponse


def test_save_skips_without_path(caplog, openai_settings):
    """Test that save() saves to default path when no explicit path is configured."""
    r = BaseResponse(
        name="test",
        instructions="hi",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    caplog.set_level("INFO")
    r.save()  # Should save to default location
    # Verify that it saved successfully to the default path
    assert any("Saved messages to" in m for m in caplog.messages)
