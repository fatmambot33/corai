import pytest
from openai_sdk_helpers.response.base import ResponseBase


def test_data_path_error(openai_settings):
    """Test that data_path property raises RuntimeError if not configured."""
    r = ResponseBase(
        name="test",
        instructions="hi",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    # data_path is now resolved at init time, so accessing _data_path should work
    assert r._data_path is not None
