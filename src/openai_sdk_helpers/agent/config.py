"""Configuration helpers for ``AgentBase``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents import Agent, Handoff, InputGuardrail, OutputGuardrail, Session
from agents.model_settings import ModelSettings

from ..utils import JSONSerializable
from ..utils.path_utils import ensure_directory


class AgentRegistry:
    """Registry for managing AgentConfig instances.

    Provides centralized storage and retrieval of agent configurations,
    enabling reusable agent specs across the application. Configurations
    are stored by name and can be retrieved or listed as needed.

    Methods
    -------
    register(config)
        Add an AgentConfig to the registry.
    get(name)
        Retrieve a configuration by name.
    list_names()
        Return all registered configuration names.
    clear()
        Remove all registered configurations.
    save_to_directory(path)
        Export all registered configurations to JSON files.

    Examples
    --------
    >>> registry = AgentRegistry()
    >>> config = AgentConfig(
    ...     name="test_agent",
    ...     model="gpt-4o-mini"
    ... )
    >>> registry.register(config)
    >>> retrieved = registry.get("test_agent")
    >>> retrieved.name
    'test_agent'
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._configs: dict[str, AgentConfig] = {}

    def register(self, config: AgentConfig) -> None:
        """Add an AgentConfig to the registry.

        Parameters
        ----------
        config : AgentConfig
            Configuration to register.

        Raises
        ------
        ValueError
            If a configuration with the same name is already registered.

        Examples
        --------
        >>> registry = AgentRegistry()
        >>> config = AgentConfig(name="test", model="gpt-4o-mini")
        >>> registry.register(config)
        """
        if config.name in self._configs:
            raise ValueError(
                f"Configuration '{config.name}' is already registered. "
                "Use a unique name or clear the registry first."
            )
        self._configs[config.name] = config

    def get(self, name: str) -> AgentConfig:
        """Retrieve a configuration by name.

        Parameters
        ----------
        name : str
            Configuration name to look up.

        Returns
        -------
        AgentConfig
            The registered configuration.

        Raises
        ------
        KeyError
            If no configuration with the given name exists.

        Examples
        --------
        >>> registry = AgentRegistry()
        >>> config = registry.get("test_agent")
        """
        if name not in self._configs:
            raise KeyError(
                f"No configuration named '{name}' found. "
                f"Available: {list(self._configs.keys())}"
            )
        return self._configs[name]

    def list_names(self) -> list[str]:
        """Return all registered configuration names.

        Returns
        -------
        list[str]
            Sorted list of configuration names.

        Examples
        --------
        >>> registry = AgentRegistry()
        >>> registry.list_names()
        []
        """
        return sorted(self._configs.keys())

    def clear(self) -> None:
        """Remove all registered configurations.

        Examples
        --------
        >>> registry = AgentRegistry()
        >>> registry.clear()
        """
        self._configs.clear()

    def save_to_directory(self, path: Path | str) -> None:
        """Export all registered configurations to JSON files in a directory.

        Serializes each registered AgentConfig to an individual JSON file
        named after the configuration. Creates the directory if it does not exist.

        Parameters
        ----------
        path : Path or str
            Directory path where JSON files will be saved. Will be created if
            it does not already exist.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If the directory cannot be created or files cannot be written.

        Examples
        --------
        >>> registry = AgentRegistry()
        >>> registry.save_to_directory("./agents")
        >>> registry.save_to_directory(Path("exports"))
        """
        dir_path = ensure_directory(Path(path))
        config_names = self.list_names()

        if not config_names:
            return

        for config_name in config_names:
            config = self.get(config_name)
            filename = f"{config_name}.json"
            filepath = dir_path / filename
            config.to_json_file(filepath)


# Global default registry instance
_default_registry = AgentRegistry()


def get_default_registry() -> AgentRegistry:
    """Return the global default registry instance.

    Returns
    -------
    AgentRegistry
        Singleton registry for application-wide configuration storage.

    Examples
    --------
    >>> registry = get_default_registry()
    >>> config = AgentConfig(name="test", model="gpt-4o-mini")
    >>> registry.register(config)
    """
    return _default_registry


@dataclass(frozen=True, slots=True)
class AgentConfig(JSONSerializable):
    """Immutable configuration for building an AgentBase.

    Encapsulates all metadata required to define an agent including its
    instructions, tools, model settings, handoffs, guardrails, and session
    management. Inherits from JSONSerializable to support serialization.

    Parameters
    ----------
    name : str
        Unique identifier for the agent. Must be a non-empty string.
    instructions : str or Path, optional
        Plain text instructions or a path to a Jinja template file whose
        contents are loaded at runtime. Default is None.
    description : str, optional
        Short description of the agent's purpose. Default is None.
    model : str, optional
        Model identifier to use (e.g., "gpt-4o-mini"). Default is None.
    template_path : str or Path, optional
        Path to the Jinja template (absolute or relative to prompt_dir).
        This takes precedence over instructions if both are provided.
        Default is None.
    input_type : type, optional
        Type describing the agent input, commonly a Pydantic model.
        Default is None.
    output_type : type, optional
        Type describing the agent output, commonly a Pydantic model or
        builtin like str. Default is None.
    tools : list, optional
        Tool definitions available to the agent. Default is None.
    model_settings : ModelSettings, optional
        Additional model configuration settings. Default is None.
    handoffs : list[Agent or Handoff], optional
        List of agents or handoff configurations that this agent can
        delegate to for specific tasks. Default is None.
    input_guardrails : list[InputGuardrail], optional
        List of guardrails to validate agent inputs before processing.
        Default is None.
    output_guardrails : list[OutputGuardrail], optional
        List of guardrails to validate agent outputs before returning.
        Default is None.
    session : Session, optional
        Session configuration for automatically maintaining conversation
        history across agent runs. Default is None.

    Methods
    -------
    __post_init__()
        Validate configuration invariants after initialization.
    instructions_text
        Return the resolved instruction content as a string.
    to_json()
        Return a JSON-compatible dict (inherited from JSONSerializable).
    to_json_file(filepath)
        Write serialized JSON data to a file (inherited from JSONSerializable).
    from_json(data)
        Create an instance from a JSON-compatible dict (inherited from JSONSerializable).
    from_json_file(filepath)
        Load an instance from a JSON file (inherited from JSONSerializable).

    Examples
    --------
    >>> config = AgentConfig(
    ...     name="summarizer",
    ...     description="Summarizes text",
    ...     model="gpt-4o-mini"
    ... )
    >>> config.name
    'summarizer'
    """

    name: str
    instructions: Optional[str | Path] = None
    description: Optional[str] = None
    model: Optional[str] = None
    template_path: Optional[str | Path] = None
    input_type: Optional[type] = None
    output_type: Optional[type] = None
    tools: Optional[list] = None
    model_settings: Optional[ModelSettings] = None
    handoffs: Optional[list[Agent | Handoff]] = None
    input_guardrails: Optional[list[InputGuardrail]] = None
    output_guardrails: Optional[list[OutputGuardrail]] = None
    session: Optional[Session] = None

    def __post_init__(self) -> None:
        """Validate configuration invariants after initialization.

        Ensures that the name is a non-empty string and that instructions
        or template_path are properly formatted if provided.

        Raises
        ------
        TypeError
            If name is not a non-empty string.
        ValueError
            If instructions is an empty string.
        FileNotFoundError
            If instructions is a Path that doesn't point to a readable file.
        """
        if not self.name or not isinstance(self.name, str):
            raise TypeError("AgentConfig.name must be a non-empty str")

        if self.instructions is not None:
            if isinstance(self.instructions, str):
                if not self.instructions.strip():
                    raise ValueError("AgentConfig.instructions must be a non-empty str")
            elif isinstance(self.instructions, Path):
                instruction_path = self.instructions.expanduser()
                if not instruction_path.is_file():
                    raise FileNotFoundError(
                        f"Instruction template not found: {instruction_path}"
                    )

        if self.template_path is not None and isinstance(self.template_path, Path):
            # Validate template_path if it's a Path object
            template = self.template_path.expanduser()
            if not template.exists():
                # We don't raise here because template_path might be relative
                # and resolved later with prompt_dir
                pass

    @property
    def instructions_text(self) -> str | None:
        """Return the resolved instruction text.

        Returns
        -------
        str or None
            Plain-text instructions, loading template files when necessary,
            or None if no instructions are configured.
        """
        return self._resolve_instructions()

    def _resolve_instructions(self) -> str | None:
        """Resolve instructions from string or file path."""
        if self.instructions is None:
            return None
        if isinstance(self.instructions, Path):
            instruction_path = self.instructions.expanduser()
            try:
                return instruction_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(
                    f"Unable to read instructions at '{instruction_path}': {exc}"
                ) from exc
        return self.instructions


__all__ = ["AgentConfig", "AgentRegistry", "get_default_registry"]
