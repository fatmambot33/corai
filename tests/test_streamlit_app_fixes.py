"""Test fixes for Streamlit app file upload and output_text display."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.streamlit_app.app import _extract_assistant_text


def test_extract_assistant_text_with_output_text() -> None:
    """Test that _extract_assistant_text extracts output_text when present."""
    # Create a mock response
    mock_response = Mock(spec=ResponseBase)

    # Create a mock message with output_text attribute
    mock_message = Mock()
    mock_message.content = Mock()
    mock_message.content.output_text = "This is the output text from the assistant"

    mock_response.get_last_assistant_message.return_value = mock_message
    mock_response.get_last_tool_message.return_value = None

    # Test that output_text is extracted
    result = _extract_assistant_text(mock_response)
    assert result == "This is the output text from the assistant"


def test_extract_assistant_text_without_output_text() -> None:
    """Test that _extract_assistant_text falls back when output_text is absent."""
    # Create a mock response
    mock_response = Mock(spec=ResponseBase)

    # Create a mock message with standard content structure
    mock_message = Mock()
    mock_message.content = Mock()
    mock_message.content.output_text = None

    # Create mock text content
    mock_text_part = Mock()
    mock_text_part.text = Mock()
    mock_text_part.text.value = "Standard text response"

    mock_message.content.content = [mock_text_part]

    mock_response.get_last_assistant_message.return_value = mock_message
    mock_response.get_last_tool_message.return_value = None

    # Test that standard content is extracted
    result = _extract_assistant_text(mock_response)
    assert result == "Standard text response"


def test_extract_assistant_text_with_no_message() -> None:
    """Test that _extract_assistant_text returns empty string when no message."""
    # Create a mock response with no messages
    mock_response = Mock(spec=ResponseBase)
    mock_response.get_last_assistant_message.return_value = None
    mock_response.get_last_tool_message.return_value = None

    # Test that empty string is returned
    result = _extract_assistant_text(mock_response)
    assert result == ""


def test_extract_assistant_text_with_multiple_text_parts() -> None:
    """Test that multiple text parts are joined correctly."""
    # Create a mock response
    mock_response = Mock(spec=ResponseBase)

    # Create a mock message with multiple text parts
    mock_message = Mock()
    mock_message.content = Mock()
    mock_message.content.output_text = None

    # Create multiple mock text parts
    mock_text_part1 = Mock()
    mock_text_part1.text = Mock()
    mock_text_part1.text.value = "First part"

    mock_text_part2 = Mock()
    mock_text_part2.text = Mock()
    mock_text_part2.text.value = "Second part"

    mock_message.content.content = [mock_text_part1, mock_text_part2]

    mock_response.get_last_assistant_message.return_value = mock_message
    mock_response.get_last_tool_message.return_value = None

    # Test that parts are joined with double newlines
    result = _extract_assistant_text(mock_response)
    assert result == "First part\n\nSecond part"


def test_extract_assistant_text_with_dict_text_structure() -> None:
    """Test extraction when content part has text as a string in a dict.

    This tests the specific case where message.content.content is a list
    of dictionaries with 'text' as a string value (not an object with .value).
    This structure appears in some OpenAI API responses.
    """
    # Create a mock response
    mock_response = Mock(spec=ResponseBase)

    # Create a mock message that matches the dict structure
    mock_message = Mock()
    mock_message.content = Mock()
    mock_message.content.output_text = None

    # The content structure where text is a string directly in the dict
    mock_content_part = {
        "annotations": [],
        "text": "Could you please provide more details or specify what you're referring to regarding the tree?",
        "type": "output_text",
        "logprobs": [],
    }

    mock_message.content.content = [mock_content_part]

    mock_response.get_last_assistant_message.return_value = mock_message
    mock_response.get_last_tool_message.return_value = None

    # Test that text content from dict structure is extracted
    result = _extract_assistant_text(mock_response)

    expected = "Could you please provide more details or specify what you're referring to regarding the tree?"
    assert result == expected
