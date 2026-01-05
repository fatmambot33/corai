"""Example demonstrating image and file data support with base64 encoding.

This example shows how to use the new image and file data features
as described in the problem statement.
"""

import base64
from pathlib import Path

from openai_sdk_helpers import OpenAISettings
from openai_sdk_helpers.response import BaseResponse
from openai_sdk_helpers.utils import encode_image, encode_file, create_image_data_url, create_file_data_url


def example_with_base64_image():
    """Example: Using base64-encoded images directly.
    
    This matches the pattern from the problem statement for images.
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
        # Method 1: Using the images parameter (recommended)
        result = response.run_sync(
            "What's in this image?",
            images="path_to_your_image.jpg"
        )
        print(result)
        
        # Method 2: Manual encoding if needed
        image_path = "path_to_your_image.jpg"
        base64_image = encode_image(image_path)
        # The encoding happens automatically when using images parameter
        

def example_with_base64_file():
    """Example: Using base64-encoded file data directly.
    
    This matches the pattern from the problem statement for files.
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
        # Using the file_data parameter (recommended)
        result = response.run_sync(
            "What is the first dragon in the book?",
            file_data="draconomicon.pdf"
        )
        print(result)


def example_with_multiple_attachments():
    """Example: Using both images and file data together."""
    settings = OpenAISettings.from_env()
    
    with BaseResponse(
        name="multi_demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Use both images and file data
        result = response.run_sync(
            "Analyze this image and document together",
            images="photo.jpg",
            file_data="document.pdf"
        )
        print(result)


def example_replacing_vector_store():
    """Example: Using base64 for attachments instead of vector stores.
    
    By default, attachments use vector stores. Set use_base64=True to
    use base64 encoding instead.
    """
    settings = OpenAISettings.from_env()
    
    with BaseResponse(
        name="base64_demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Use base64 for attachments instead of vector stores
        result = response.run_sync(
            "Analyze this document",
            attachments="document.pdf",
            use_base64=True  # Force base64 encoding
        )
        print(result)


def example_from_problem_statement_images():
    """Example directly from the problem statement - images."""
    from openai import OpenAI
    from openai_sdk_helpers.utils import encode_image
    
    client = OpenAI()

    # Path to your image
    image_path = "path_to_your_image.jpg"

    # Getting the Base64 string
    base64_image = encode_image(image_path)

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
            images=image_path  # Handles encoding automatically
        )
        print(result)


def example_from_problem_statement_files():
    """Example directly from the problem statement - files."""
    from openai import OpenAI
    from openai_sdk_helpers.utils import encode_file
    
    client = OpenAI()

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
            file_data="draconomicon.pdf"  # Handles encoding automatically
        )
        print(result)


if __name__ == "__main__":
    print("See function examples above for usage patterns.")
    print("Note: These examples require valid file paths and an OpenAI API key.")
