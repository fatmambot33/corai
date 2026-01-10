"""Tests for new AgentConfiguration and AgentConfigurationRegistry improvements."""

from __future__ import annotations

from pathlib import Path

import pytest

from openai_sdk_helpers.agent.config import (
    AgentConfiguration,
    AgentConfigurationRegistry,
)


def test_agent_config_replace_method() -> None:
    """Test that AgentConfiguration.replace creates a new instance with changes."""
    original = AgentConfiguration(
        name="agent1",
        instructions="Test instructions",
        model="gpt-4o-mini",
        description="Original description",
    )

    # Replace single field
    modified = original.replace(name="agent2")
    assert modified.name == "agent2"
    assert modified.model == "gpt-4o-mini"
    assert modified.description == "Original description"

    # Replace multiple fields
    modified2 = original.replace(name="agent3", description="New description")
    assert modified2.name == "agent3"
    assert modified2.description == "New description"
    assert modified2.model == "gpt-4o-mini"

    # Original should be unchanged
    assert original.name == "agent1"
    assert original.description == "Original description"


def test_agent_config_to_agent_base() -> None:
    """Test that AgentConfiguration.create_agent creates a BaseAgent instance."""
    config = AgentConfiguration(
        name="test_agent", model="gpt-4o-mini", instructions="Test instructions"
    )
    agent = config.create_agent()

    assert agent.agent_name == "test_agent"
    assert agent.model == "gpt-4o-mini"


def test_agent_registry_load_from_directory(tmp_path: Path) -> None:
    """Test loading configurations from a directory."""
    # Create some config files
    config1 = AgentConfiguration(
        name="agent1",
        model="gpt-4o-mini",
        description="First",
        instructions="Test instructions",
    )
    config2 = AgentConfiguration(
        name="agent2",
        model="gpt-4",
        description="Second",
        instructions="Test instructions",
    )

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # Save directly in the directory (not subdirectory)
    config1.to_json_file(configs_dir / "agent1.json")
    config2.to_json_file(configs_dir / "agent2.json")

    # Load into registry
    registry = AgentConfigurationRegistry()
    count = registry.load_from_directory(configs_dir)

    assert count == 2
    assert len(registry.list_names()) == 2
    assert "agent1" in registry.list_names()
    assert "agent2" in registry.list_names()

    loaded1 = registry.get("agent1")
    assert loaded1.name == "agent1"
    assert loaded1.description == "First"

    loaded2 = registry.get("agent2")
    assert loaded2.name == "agent2"
    assert loaded2.description == "Second"


def test_agent_registry_load_from_directory_not_found() -> None:
    """Test that loading from non-existent directory raises error."""
    registry = AgentConfigurationRegistry()

    with pytest.raises(FileNotFoundError):
        registry.load_from_directory("/nonexistent/directory")


def test_agent_registry_load_from_directory_not_a_dir(tmp_path: Path) -> None:
    """Test that loading from a file (not directory) raises error."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("test")

    registry = AgentConfigurationRegistry()

    with pytest.raises(NotADirectoryError):
        registry.load_from_directory(file_path)


def test_agent_registry_load_from_directory_invalid_json(tmp_path: Path) -> None:
    """Test that invalid JSON files are skipped with warning."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # Create valid config
    config = AgentConfiguration(
        name="valid", model="gpt-4o-mini", instructions="Test instructions"
    )
    config.to_json_file(configs_dir / "valid.json")

    # Create invalid JSON file
    (configs_dir / "invalid.json").write_text("not valid json {")

    # Should load the valid one and warn about the invalid one
    registry = AgentConfigurationRegistry()
    with pytest.warns(UserWarning, match="Failed to load configuration"):
        count = registry.load_from_directory(configs_dir)

    assert count == 1
    assert "valid" in registry.list_names()


def test_agent_registry_save_and_load_round_trip(tmp_path: Path) -> None:
    """Test saving and loading configurations maintains data integrity."""
    # Create registry with configs
    registry1 = AgentConfigurationRegistry()
    config1 = AgentConfiguration(
        name="agent1",
        model="gpt-4o-mini",
        description="Test 1",
        instructions="Test instructions",
    )
    config2 = AgentConfiguration(
        name="agent2",
        model="gpt-4",
        description="Test 2",
        instructions="Test instructions",
    )
    registry1.register(config1)
    registry1.register(config2)

    # Save to directory (files will be in subdirectory)
    save_dir = tmp_path / "agents"
    registry1.save_to_directory(save_dir)

    # Move files from subdirectory to main directory so load_from_directory can find them
    subdir = save_dir / "AgentConfiguration"
    for f in subdir.glob("*.json"):
        f.rename(save_dir / f.name)

    # Load into new registry
    registry2 = AgentConfigurationRegistry()
    count = registry2.load_from_directory(save_dir)

    assert count == 2
    assert registry2.list_names() == registry1.list_names()

    # Verify data integrity
    for name in registry1.list_names():
        orig = registry1.get(name)
        loaded = registry2.get(name)
        assert loaded.name == orig.name
        assert loaded.model == orig.model
        assert loaded.description == orig.description
