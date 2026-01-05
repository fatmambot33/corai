"""Utilities for encoding files and images to base64.

This module provides helper functions for encoding images and files
to base64 format for use with OpenAI API's input_image and input_file
content types.
"""

import base64
import mimetypes
from pathlib import Path
from typing import Literal


def encode_image(image_path: str | Path) -> str:
    """Encode an image file to base64.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file to encode.

    Returns
    -------
    str
        Base64-encoded string representation of the image.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    IOError
        If the file cannot be read.

    Examples
    --------
    >>> base64_image = encode_image("photo.jpg")
    >>> image_url = f"data:image/jpeg;base64,{base64_image}"
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_file(file_path: str | Path) -> str:
    """Encode a file to base64.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to encode.

    Returns
    -------
    str
        Base64-encoded string representation of the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    IOError
        If the file cannot be read.

    Examples
    --------
    >>> base64_file = encode_file("document.pdf")
    >>> file_data = f"data:application/pdf;base64,{base64_file}"
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(file_path: str | Path) -> str:
    """Get the MIME type of a file.

    Parameters
    ----------
    file_path : str or Path
        Path to the file.

    Returns
    -------
    str
        MIME type of the file, or "application/octet-stream" if unknown.

    Examples
    --------
    >>> get_mime_type("photo.jpg")
    'image/jpeg'
    >>> get_mime_type("document.pdf")
    'application/pdf'
    """
    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def create_image_data_url(
    image_path: str | Path, detail: Literal["low", "high", "auto"] = "auto"
) -> tuple[str, Literal["low", "high", "auto"]]:
    """Create a data URL for an image with MIME type detection.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    detail : {"low", "high", "auto"}, default "auto"
        Detail level for image processing.

    Returns
    -------
    tuple[str, str]
        A tuple containing the data URL and detail level.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    Examples
    --------
    >>> image_url, detail = create_image_data_url("photo.jpg", "high")
    >>> # Use with ResponseInputImageContentParam
    """
    mime_type = get_mime_type(image_path)
    base64_image = encode_image(image_path)
    data_url = f"data:{mime_type};base64,{base64_image}"
    return data_url, detail


def create_file_data_url(file_path: str | Path) -> str:
    """Create a data URL for a file with MIME type detection.

    Parameters
    ----------
    file_path : str or Path
        Path to the file.

    Returns
    -------
    str
        Data URL with MIME type and base64-encoded content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Examples
    --------
    >>> file_data = create_file_data_url("document.pdf")
    >>> # Use with ResponseInputFileContentParam
    """
    mime_type = get_mime_type(file_path)
    base64_file = encode_file(file_path)
    return f"data:{mime_type};base64,{base64_file}"
