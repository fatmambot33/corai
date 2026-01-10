"""Tests for the AgentConfigurationRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest

from openai_sdk_helpers.agent.config import (
    AgentConfiguration,
    AgentConfigurationRegistry,
    get_default_registry,
)


def test_agent_registry_basic_operations() -> None:
    """Test basic registry operations."""
    registry = AgentConfigurationRegistry()
    config = AgentConfiguration(
        name="test_agent", model="gpt-4o-mini", instructions="Test instructions"
    )

    # Register
    registry.register(config)
    assert "test_agent" in registry.list_names()

    # Get
    retrieved = registry.get("test_agent")
    assert retrieved.name == "test_agent"
    assert retrieved.model == "gpt-4o-mini"

    # Clear
    registry.clear()
    assert len(registry.list_names()) == 0


def test_agent_registry_duplicate_name_raises() -> None:
    """Test that registering duplicate names raises ValueError."""
    registry = AgentConfigurationRegistry()
    config1 = AgentConfiguration(
        name="duplicate", model="gpt-4o-mini", instructions="Test instructions"
    )
    config2 = AgentConfiguration(
        name="duplicate", model="gpt-4", instructions="Test instructions"
    )

    registry.register(config1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(config2)


def test_agent_registry_get_nonexistent_raises() -> None:
    """Test that getting a nonexistent config raises KeyError."""
    registry = AgentConfigurationRegistry()

    with pytest.raises(KeyError, match="No configuration named"):
        registry.get("nonexistent")


def test_agent_registry_clear() -> None:
    """Test clearing the registry."""
    registry = AgentConfigurationRegistry()
    config1 = AgentConfiguration(
        name="agent1", model="gpt-4o-mini", instructions="Test instructions"
    )
    config2 = AgentConfiguration(
        name="agent2", model="gpt-4", instructions="Test instructions"
    )

    registry.register(config1)
    registry.register(config2)
    assert len(registry.list_names()) == 2

    registry.clear()
    assert len(registry.list_names()) == 0


def test_agent_registry_multiple_configs() -> None:
    """Test registering and retrieving multiple configurations."""
    registry = AgentConfigurationRegistry()
    configs = [
        AgentConfiguration(
            name=f"agent{i}", model="gpt-4o-mini", instructions="Test instructions"
        )
        for i in range(5)
    ]

    for config in configs:
        registry.register(config)

    names = registry.list_names()
    assert len(names) == 5
    assert names == ["agent0", "agent1", "agent2", "agent3", "agent4"]

    for i, config in enumerate(configs):
        retrieved = registry.get(f"agent{i}")
        assert retrieved.name == config.name


def test_get_default_registry() -> None:
    """Test that get_default_registry returns a singleton."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()

    assert registry1 is registry2


def test_agent_registry_isolated_instances() -> None:
    """Test that separate registry instances are isolated."""
    registry1 = AgentConfigurationRegistry()
    registry2 = AgentConfigurationRegistry()

    config = AgentConfiguration(
        name="test", model="gpt-4o-mini", instructions="Test instructions"
    )
    registry1.register(config)

    assert "test" in registry1.list_names()
    assert "test" not in registry2.list_names()


def test_agent_registry_save_to_directory(tmp_path: Path) -> None:
    """Test saving registry to directory."""
    registry = AgentConfigurationRegistry()
    config1 = AgentConfiguration(
        name="agent1",
        model="gpt-4o-mini",
        description="First agent",
        instructions="Test instructions",
    )
    config2 = AgentConfiguration(
        name="agent2",
        model="gpt-4",
        description="Second agent",
        instructions="Test instructions",
    )

    registry.register(config1)
    registry.register(config2)

    save_dir = tmp_path / "agents"
    registry.save_to_directory(save_dir)

    subdir = save_dir / "AgentConfiguration"
    assert subdir.exists()
    assert (subdir / "agent1.json").exists()
    assert (subdir / "agent2.json").exists()

    # Verify content
    loaded_config = AgentConfiguration.from_json_file(subdir / "agent1.json")
    assert loaded_config.name == "agent1"
    assert loaded_config.description == "First agent"


def test_agent_registry_save_empty_directory(tmp_path: Path) -> None:
    """Test saving empty registry creates directory but no files."""
    registry = AgentConfigurationRegistry()
    save_dir = tmp_path / "empty_agents"

    registry.save_to_directory(save_dir)

    # Directory should be created even if empty
    assert save_dir.exists()
    # But no files should be created
    assert len(list(save_dir.glob("*.json"))) == 0
