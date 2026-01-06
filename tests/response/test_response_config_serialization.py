"""Tests for ResponseConfiguration JSON serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openai_sdk_helpers.response.config import ResponseConfiguration


def test_response_config_to_json() -> None:
    """Test that ResponseConfiguration can be serialized to JSON."""
    config = ResponseConfiguration(
        name="test_config",
        instructions="Test instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        system_vector_store=["store1", "store2"],
        data_path=Path("/tmp/test_data"),
    )

    json_data = config.to_json()

    assert isinstance(json_data, dict)
    assert json_data["name"] == "test_config"
    assert json_data["instructions"] == "Test instructions"
    assert json_data["tools"] is None
    assert json_data["input_structure"] is None
    assert json_data["output_structure"] is None
    assert json_data["system_vector_store"] == ["store1", "store2"]
    # Path should be converted to string
    assert json_data["data_path"] == "/tmp/test_data"


def test_response_config_to_json_file(tmp_path: Path) -> None:
    """Test that ResponseConfiguration can be written to a JSON file."""
    config = ResponseConfiguration(
        name="test_config",
        instructions="Test instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        system_vector_store=["store1"],
    )

    json_file = tmp_path / "config.json"
    result_path = config.to_json_file(json_file)

    assert result_path == str(json_file)
    assert json_file.exists()

    # Verify the file content
    with open(json_file, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    assert loaded_data["name"] == "test_config"
    assert loaded_data["instructions"] == "Test instructions"
    assert loaded_data["system_vector_store"] == ["store1"]


def test_response_config_json_serialization_with_none_fields() -> None:
    """Test JSON serialization with None fields."""
    config = ResponseConfiguration(
        name="minimal_config",
        instructions="Minimal instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    json_data = config.to_json()

    assert json_data["name"] == "minimal_config"
    assert json_data["system_vector_store"] is None
    assert json_data["data_path"] is None
