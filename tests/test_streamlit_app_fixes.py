"""Test fixes for Streamlit app file upload and output_text display."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from openai_sdk_helpers.response.base import BaseResponse
from openai_sdk_helpers.streamlit_app.app import _extract_assistant_text


def test_extract_assistant_text_with_output_text() -> None:
    """Test that _extract_assistant_text extracts output_text when present."""
    # Create a mock response
    mock_response = Mock(spec=BaseResponse)

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
    mock_response = Mock(spec=BaseResponse)

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
    mock_response = Mock(spec=BaseResponse)
    mock_response.get_last_assistant_message.return_value = None
    mock_response.get_last_tool_message.return_value = None

    # Test that empty string is returned
    result = _extract_assistant_text(mock_response)
    assert result == ""


def test_extract_assistant_text_with_multiple_text_parts() -> None:
    """Test that multiple text parts are joined correctly."""
    # Create a mock response
    mock_response = Mock(spec=BaseResponse)

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
