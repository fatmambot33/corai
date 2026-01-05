"""Tests for BaseResponse with image and file data support."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openai_sdk_helpers.response.base import BaseResponse


@pytest.fixture
def response_base(openai_settings, tmp_path):
    """Return a BaseResponse instance for testing."""
    return BaseResponse(
        name="test",
        instructions="test instructions",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
        data_path=tmp_path,
    )


def test_build_input_with_images(response_base, tmp_path):
    """Test _build_input with image attachments."""
    # Create a temporary image file
    image_path = tmp_path / "test_image.jpg"
    image_content = b"fake image content"
    image_path.write_bytes(image_content)

    # Build input with image
    response_base._build_input(
        content="What's in this image?",
        images=[str(image_path)]
    )

    # Verify message was added
    assert len(response_base.messages.messages) == 2  # system + user
    user_message = response_base.messages.messages[-1]
    assert user_message.role == "user"
    
    # Check that the message content has an image
    content = user_message.content["content"]
    assert len(content) == 2  # text + image
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert "base64" in content[1]["image_url"]


def test_build_input_with_file_data(response_base, tmp_path):
    """Test _build_input with file data attachments."""
    # Create a temporary file
    file_path = tmp_path / "test_file.pdf"
    file_content = b"fake pdf content"
    file_path.write_bytes(file_content)

    # Build input with file data
    response_base._build_input(
        content="Analyze this document",
        file_data=[str(file_path)]
    )

    # Verify message was added
    assert len(response_base.messages.messages) == 2  # system + user
    user_message = response_base.messages.messages[-1]
    assert user_message.role == "user"
    
    # Check that the message content has a file
    content = user_message.content["content"]
    assert len(content) == 2  # text + file
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_file"
    assert "file_data" in content[1]
    assert "base64" in content[1]["file_data"]
    assert content[1]["filename"] == "test_file.pdf"


def test_build_input_with_base64_attachments(response_base, tmp_path):
    """Test _build_input with use_base64 flag."""
    # Create a temporary file
    file_path = tmp_path / "test_file.txt"
    file_content = b"test content"
    file_path.write_bytes(file_content)

    # Build input with base64 flag
    response_base._build_input(
        content="Process this file",
        attachments=[str(file_path)],
        use_base64=True
    )

    # Verify message was added
    assert len(response_base.messages.messages) == 2  # system + user
    user_message = response_base.messages.messages[-1]
    assert user_message.role == "user"
    
    # Check that the message content has a file with base64 data
    content = user_message.content["content"]
    assert len(content) == 2  # text + file
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_file"
    assert "file_data" in content[1]
    assert "base64" in content[1]["file_data"]


def test_build_input_with_multiple_types(response_base, tmp_path):
    """Test _build_input with multiple attachment types."""
    # Create temporary files
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(b"fake image")
    
    file_path = tmp_path / "test_doc.pdf"
    file_path.write_bytes(b"fake pdf")

    # Build input with multiple types
    response_base._build_input(
        content="Analyze these",
        images=[str(image_path)],
        file_data=[str(file_path)]
    )

    # Verify message was added
    user_message = response_base.messages.messages[-1]
    content = user_message.content["content"]
    
    # Should have text + image + file
    assert len(content) == 3
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_file"
    assert content[2]["type"] == "input_image"


def test_build_input_vector_store_still_works(response_base, tmp_path):
    """Test that vector store attachment still works without use_base64."""
    # Create a temporary file
    file_path = tmp_path / "test_file.txt"
    file_path.write_bytes(b"test content")

    # Mock the vector storage
    mock_storage = MagicMock()
    mock_file = MagicMock()
    mock_file.id = "file_123"
    mock_storage.upload_file.return_value = mock_file
    mock_storage.id = "vs_123"
    
    with patch("openai_sdk_helpers.vector_storage.VectorStorage", return_value=mock_storage):
        response_base._build_input(
            content="Process this file",
            attachments=[str(file_path)],
            use_base64=False
        )

    # Verify vector storage was used
    mock_storage.upload_file.assert_called_once()
    
    # Verify message uses file_id
    user_message = response_base.messages.messages[-1]
    content = user_message.content["content"]
    assert content[1]["type"] == "input_file"
    assert content[1]["file_id"] == "file_123"


def test_run_sync_with_images(response_base, tmp_path):
    """Test run_sync method signature accepts images parameter."""
    # Create a temporary image file
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image")

    # We just need to verify the method signature accepts the parameter
    # Don't actually call run_sync as it involves complex async behavior
    from inspect import signature
    sig = signature(response_base.run_sync)
    assert 'images' in sig.parameters
    assert 'file_data' in sig.parameters
    assert 'use_base64' in sig.parameters


def test_run_async_with_images(response_base, tmp_path):
    """Test run_async method signature accepts images parameter."""
    # Create a temporary image file
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image")

    # We just need to verify the method signature accepts the parameter
    from inspect import signature
    sig = signature(response_base.run_async)
    assert 'images' in sig.parameters
    assert 'file_data' in sig.parameters
    assert 'use_base64' in sig.parameters
