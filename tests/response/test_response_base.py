"""Tests for the ResponseBase class."""

from openai_sdk_helpers.response.base import ResponseBase


def test_response_base_initialization(openai_settings, mock_openai_client):
    """Test the initialization of the ResponseBase class."""
    instance = ResponseBase(
        name="test",
        instructions="Test instructions",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    assert instance._instructions == "Test instructions"
    assert instance._model == "gpt-4o-mini"
    assert instance.messages.messages[0].role == "system"
    assert instance._client is mock_openai_client
