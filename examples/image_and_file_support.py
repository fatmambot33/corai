"""Example demonstrating automatic file type detection with base64 encoding.

This example shows how to use the simplified files API with automatic
type detection for images and documents.
"""

from openai_sdk_helpers import OpenAISettings
from openai_sdk_helpers.response import BaseResponse
from openai_sdk_helpers.utils import (
    encode_image,
)


def example_with_image():
    """Example: Analyze an image using automatic type detection.

    Images are automatically detected and sent as base64-encoded images.
    """
    # Initialize settings and response
    settings = OpenAISettings.from_env()

    with BaseResponse(
        name="image_demo",
        instructions="You are a helpful assistant that can analyze images.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Single file - automatically detected as image
        result = response.run_sync(
            "What's in this image?", files="path_to_your_image.jpg"
        )
        print(result)

        # Manual encoding example (for reference)
        image_path = "path_to_your_image.jpg"
        # You can manually encode if needed, but the files parameter handles it
        manual_encoded = encode_image(image_path)
        print(f"Manually encoded (first 80 chars): {manual_encoded[:80]}...")


def example_with_document():
    """Example: Analyze a document using automatic type detection.

    Documents are automatically detected and sent as base64-encoded files.
    """
    # Initialize settings and response
    settings = OpenAISettings.from_env()

    with BaseResponse(
        name="file_demo",
        instructions="You are a helpful assistant that can analyze documents.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Single document - automatically detected
        result = response.run_sync(
            "What is the first dragon in the book?", files="draconomicon.pdf"
        )
        print(result)


def example_with_multiple_files():
    """Example: Process both images and documents together.

    The files parameter accepts a list and automatically detects each file type.
    """
    settings = OpenAISettings.from_env()

    with BaseResponse(
        name="multi_demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Mix images and documents - automatic detection!
        result = response.run_sync(
            "Analyze this image and document together",
            files=["photo.jpg", "document.pdf"],
        )
        print(result)


def example_using_vector_store():
    """Example: Using vector stores for documents instead of base64.

    By default, files use base64 encoding. Set use_vector_store=True to
    store documents in a vector store for RAG capabilities.
    """
    settings = OpenAISettings.from_env()

    with BaseResponse(
        name="vector_store_demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Use vector stores for documents (enables RAG)
        result = response.run_sync(
            "Search and analyze this document",
            files="document.pdf",
            use_vector_store=True,  # Enable vector store for RAG
        )
        print(result)


def example_from_problem_statement_images():
    """Example directly from the problem statement - images."""
    # With openai-sdk-helpers, this is simplified:
    settings = OpenAISettings.from_env()

    with BaseResponse(
        name="demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        result = response.run_sync(
            "what's in this image?",
            files="path_to_your_image.jpg",  # Automatically detected and encoded
        )
        print(result)


def example_from_problem_statement_files():
    """Example directly from the problem statement - files."""
    # With openai-sdk-helpers, file encoding is simplified:
    settings = OpenAISettings.from_env()

    with BaseResponse(
        name="demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        result = response.run_sync(
            "What is the first dragon in the book?",
            files="draconomicon.pdf",  # Automatically encoded
        )
        print(result)


if __name__ == "__main__":
    print("See function examples above for usage patterns.")
    print("Note: These examples require valid file paths and an OpenAI API key.")
