"""Unit tests for encoding utilities."""

import base64
from unittest.mock import mock_open, patch

import pytest

from openai_sdk_helpers.utils.encoding import (
    create_file_data_url,
    create_image_data_url,
    encode_file,
    encode_image,
    get_mime_type,
    is_image_file,
)


def test_encode_image_success(tmp_path):
    """Test encoding an image file to base64."""
    # Create a temporary image file
    image_path = tmp_path / "test_image.jpg"
    image_content = b"fake image content"
    image_path.write_bytes(image_content)

    # Encode the image
    result = encode_image(str(image_path))

    # Verify the result
    expected = base64.b64encode(image_content).decode("utf-8")
    assert result == expected


def test_encode_image_not_found():
    """Test encoding a non-existent image file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        encode_image("nonexistent.jpg")


def test_encode_file_success(tmp_path):
    """Test encoding a file to base64."""
    # Create a temporary file
    file_path = tmp_path / "test_file.pdf"
    file_content = b"fake pdf content"
    file_path.write_bytes(file_content)

    # Encode the file
    result = encode_file(str(file_path))

    # Verify the result
    expected = base64.b64encode(file_content).decode("utf-8")
    assert result == expected


def test_encode_file_not_found():
    """Test encoding a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        encode_file("nonexistent.pdf")


def test_get_mime_type_known_types():
    """Test MIME type detection for known file types."""
    assert get_mime_type("photo.jpg") == "image/jpeg"
    assert get_mime_type("photo.jpeg") == "image/jpeg"
    assert get_mime_type("document.pdf") == "application/pdf"
    assert get_mime_type("image.png") == "image/png"
    assert get_mime_type("page.html") == "text/html"


def test_get_mime_type_unknown():
    """Test MIME type detection for unknown file types."""
    result = get_mime_type("unknown.unknownext123")
    assert result == "application/octet-stream"


def test_create_image_data_url(tmp_path):
    """Test creating a data URL for an image."""
    # Create a temporary image file
    image_path = tmp_path / "test_image.png"
    image_content = b"fake image content"
    image_path.write_bytes(image_content)

    # Create data URL
    data_url, detail = create_image_data_url(str(image_path), detail="high")

    # Verify the result
    expected_base64 = base64.b64encode(image_content).decode("utf-8")
    assert data_url == f"data:image/png;base64,{expected_base64}"
    assert detail == "high"


def test_create_image_data_url_default_detail(tmp_path):
    """Test creating a data URL with default detail level."""
    # Create a temporary image file
    image_path = tmp_path / "test_image.jpg"
    image_content = b"fake image content"
    image_path.write_bytes(image_content)

    # Create data URL with default detail
    data_url, detail = create_image_data_url(str(image_path))

    # Verify the result
    expected_base64 = base64.b64encode(image_content).decode("utf-8")
    assert data_url == f"data:image/jpeg;base64,{expected_base64}"
    assert detail == "auto"


def test_create_file_data_url(tmp_path):
    """Test creating a data URL for a file."""
    # Create a temporary file
    file_path = tmp_path / "test_document.pdf"
    file_content = b"fake pdf content"
    file_path.write_bytes(file_content)

    # Create data URL
    data_url = create_file_data_url(str(file_path))

    # Verify the result
    expected_base64 = base64.b64encode(file_content).decode("utf-8")
    assert data_url == f"data:application/pdf;base64,{expected_base64}"


def test_create_image_data_url_not_found():
    """Test creating a data URL for a non-existent image raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        create_image_data_url("nonexistent.jpg")


def test_create_file_data_url_not_found():
    """Test creating a data URL for a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        create_file_data_url("nonexistent.pdf")


def test_encode_with_path_object(tmp_path):
    """Test encoding functions work with Path objects."""
    # Create a temporary file
    file_path = tmp_path / "test_file.txt"
    file_content = b"test content"
    file_path.write_bytes(file_content)

    # Test with Path object
    result = encode_file(file_path)
    expected = base64.b64encode(file_content).decode("utf-8")
    assert result == expected


def test_is_image_file():
    """Test image file detection."""
    # Test image files
    assert is_image_file("photo.jpg") is True
    assert is_image_file("photo.jpeg") is True
    assert is_image_file("image.png") is True
    assert is_image_file("graphic.gif") is True
    assert is_image_file("picture.bmp") is True

    # Test non-image files
    assert is_image_file("document.pdf") is False
    assert is_image_file("text.txt") is False
    assert is_image_file("data.csv") is False
    assert is_image_file("page.html") is False
