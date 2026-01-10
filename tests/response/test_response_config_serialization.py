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
        add_output_instructions=False,
    )

    json_data = config.to_json()

    assert isinstance(json_data, dict)
    assert json_data["name"] == "test_config"
    assert json_data["instructions"] == "Test instructions"
    assert json_data["tools"] is None
    assert json_data["input_structure"] is None
    assert json_data["output_structure"] is None
    assert json_data["system_vector_store"] == ["store1", "store2"]
    assert json_data["add_output_instructions"] is False


def test_response_config_to_json_file(tmp_path: Path) -> None:
    """Test that ResponseConfiguration can be written to a JSON file."""
    config = ResponseConfiguration(
        name="test_config",
        instructions="Test instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        system_vector_store=["store1"],
        add_output_instructions=True,
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
    assert json_data["add_output_instructions"] is True


def test_response_config_from_json() -> None:
    """Test that ResponseConfiguration can be deserialized from JSON."""
    json_data = {
        "name": "test_config",
        "instructions": "Test instructions",
        "tools": None,
        "input_structure": None,
        "output_structure": None,
        "system_vector_store": ["store1", "store2"],
        "add_output_instructions": False,
    }

    config = ResponseConfiguration.from_json(json_data)

    assert config.name == "test_config"
    assert config.instructions == "Test instructions"
    assert config.tools is None
    assert config.input_structure is None
    assert config.output_structure is None
    assert config.system_vector_store == ["store1", "store2"]
    assert config.add_output_instructions is False


def test_response_config_from_json_file(tmp_path: Path) -> None:
    """Test that ResponseConfiguration can be loaded from a JSON file."""
    json_data = {
        "name": "test_config",
        "instructions": "Test instructions",
        "tools": None,
        "input_structure": None,
        "output_structure": None,
        "system_vector_store": ["store1"],
        "add_output_instructions": True,
    }

    json_file = tmp_path / "config.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    config = ResponseConfiguration.from_json_file(json_file)

    assert config.name == "test_config"
    assert config.instructions == "Test instructions"
    assert config.system_vector_store == ["store1"]
    assert config.add_output_instructions is True


def test_response_config_round_trip_serialization() -> None:
    """Test that serialization and deserialization preserve data."""
    original_config = ResponseConfiguration(
        name="round_trip_test",
        instructions="Round trip instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        system_vector_store=["store1", "store2"],
        add_output_instructions=False,
    )

    # Serialize to JSON
    json_data = original_config.to_json()

    # Deserialize from JSON
    restored_config = ResponseConfiguration.from_json(json_data)

    # Verify all fields match
    assert restored_config.name == original_config.name
    assert restored_config.instructions == original_config.instructions
    assert restored_config.tools == original_config.tools
    assert restored_config.input_structure == original_config.input_structure
    assert restored_config.output_structure == original_config.output_structure
    assert restored_config.system_vector_store == original_config.system_vector_store
    assert (
        restored_config.add_output_instructions
        == original_config.add_output_instructions
    )


def test_response_config_from_json_file_not_found() -> None:
    """Test that from_json_file raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        ResponseConfiguration.from_json_file("/nonexistent/file.json")
