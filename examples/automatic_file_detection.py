"""Example demonstrating automatic file type detection.

This example shows how to use the simplified API where you provide
a single list of files and the system automatically detects whether
they are images or documents.
"""

from openai_sdk_helpers import OpenAISettings
from openai_sdk_helpers.response import ResponseBase


def example_automatic_detection():
    """Demonstrate automatic file type detection.

    Images and documents are automatically handled based on MIME type.
    """
    settings = OpenAISettings.from_env()

    with ResponseBase(
        name="auto_demo",
        instructions="You are a helpful assistant that can analyze files.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Automatic detection - images as base64, documents inline
        result = response.run_sync(
            "Analyze these files", files=["photo.jpg", "document.pdf"]
        )
        print(result)


def example_with_single_image():
    """Analyze a single image.

    Images are automatically detected and sent as base64-encoded images.
    """
    settings = OpenAISettings.from_env()

    with ResponseBase(
        name="image_demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        result = response.run_sync(
            "What's in this image?",
            files="photo.jpg",  # Automatically detected as image
        )
        print(result)


def example_with_single_document():
    """Analyze a single document.

    Non-image files are sent as base64-encoded file data by default.
    """
    settings = OpenAISettings.from_env()

    with ResponseBase(
        name="doc_demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        result = response.run_sync(
            "What is the main topic of this document?",
            files="document.pdf",  # Automatically detected as document
        )
        print(result)


def example_with_vector_store():
    """Use a vector store for RAG.

    Documents can optionally be uploaded to a vector store for
    retrieval-augmented generation (RAG).
    """
    settings = OpenAISettings.from_env()

    with ResponseBase(
        name="rag_demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Use vector store for document search
        result = response.run_sync(
            "What is the first dragon mentioned in these books?",
            files=["draconomicon.pdf", "monster_manual.pdf"],
            use_vector_store=True,  # Enable RAG
        )
        print(result)


def example_mixed_files():
    """Process a mix of images and documents.

    All files are provided in one list. The system automatically
    categorizes them and handles them appropriately.
    """
    settings = OpenAISettings.from_env()

    with ResponseBase(
        name="mixed_demo",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        # Mix of images and documents
        result = response.run_sync(
            "Compare the chart in the image with the data in the spreadsheet",
            files=[
                "chart.png",  # Image - base64 encoded
                "data.xlsx",  # Document - base64 encoded
                "photo.jpg",  # Image - base64 encoded
                "report.pdf",  # Document - base64 encoded
            ],
        )
        print(result)


def example_from_problem_statement():
    """Run examples directly from the problem statement.

    Simplified to use automatic type detection.
    """
    settings = OpenAISettings.from_env()

    # Example 1: Image analysis
    with ResponseBase(
        name="vision",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        result = response.run_sync(
            "what's in this image?", files="path_to_your_image.jpg"
        )
        print(result)

    # Example 2: Document analysis
    with ResponseBase(
        name="doc_analysis",
        instructions="You are a helpful assistant.",
        tools=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
    ) as response:
        result = response.run_sync(
            "What is the first dragon in the book?", files="draconomicon.pdf"
        )
        print(result)


if __name__ == "__main__":
    print("See function examples above for usage patterns.")
    print("Note: These examples require valid file paths and an OpenAI API key.")
