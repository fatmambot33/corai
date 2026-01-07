"""Configuration helpers for ``BaseAgent``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents import Agent, Handoff, InputGuardrail, OutputGuardrail, Session
from agents.model_settings import ModelSettings

from ..utils import JSONSerializable
from ..utils.path_utils import ensure_directory


class AgentConfigurationRegistry:
    """Registry for managing AgentConfiguration instances.

    Provides centralized storage and retrieval of agent configurations,
    enabling reusable agent specs across the application. Configurations
    are stored by name and can be retrieved or listed as needed.

    Methods
    -------
    register(config)
        Add an AgentConfiguration to the registry.
    get(name)
        Retrieve a configuration by name.
    list_names()
        Return all registered configuration names.
    clear()
        Remove all registered configurations.
    save_to_directory(path)
        Export all registered configurations to JSON files.
    load_from_directory(path)
        Load configurations from JSON files in a directory.

    Examples
    --------
    >>> registry = AgentConfigurationRegistry()
    >>> config = AgentConfiguration(
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
        self._configs: dict[str, AgentConfiguration] = {}

    def register(self, config: AgentConfiguration) -> None:
        """Add an AgentConfiguration to the registry.

        Parameters
        ----------
        config : AgentConfiguration
            Configuration to register.

        Raises
        ------
        ValueError
            If a configuration with the same name is already registered.

        Examples
        --------
        >>> registry = AgentConfigurationRegistry()
        >>> config = AgentConfiguration(name="test", model="gpt-4o-mini")
        >>> registry.register(config)
        """
        if config.name in self._configs:
            raise ValueError(
                f"Configuration '{config.name}' is already registered. "
                "Use a unique name or clear the registry first."
            )
        self._configs[config.name] = config

    def get(self, name: str) -> AgentConfiguration:
        """Retrieve a configuration by name.

        Parameters
        ----------
        name : str
            Configuration name to look up.

        Returns
        -------
        AgentConfiguration
            The registered configuration.

        Raises
        ------
        KeyError
            If no configuration with the given name exists.

        Examples
        --------
        >>> registry = AgentConfigurationRegistry()
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
        >>> registry = AgentConfigurationRegistry()
        >>> registry.list_names()
        []
        """
        return sorted(self._configs.keys())

    def clear(self) -> None:
        """Remove all registered configurations.

        Examples
        --------
        >>> registry = AgentConfigurationRegistry()
        >>> registry.clear()
        """
        self._configs.clear()

    def save_to_directory(self, path: Path | str) -> None:
        """Export all registered configurations to JSON files in a directory.

        Serializes each registered AgentConfiguration to an individual JSON file
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
        >>> registry = AgentConfigurationRegistry()
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

    def load_from_directory(self, path: Path | str) -> int:
        """Load all agent configurations from JSON files in a directory.

        Scans the directory for JSON files and attempts to load each as an
        AgentConfiguration. Successfully loaded configurations are registered.
        Existing configurations with the same name will cause a ValueError.

        Parameters
        ----------
        path : Path or str
            Directory path containing JSON configuration files.

        Returns
        -------
        int
            Number of configurations successfully loaded and registered.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        ValueError
            If a configuration with the same name is already registered.

        Examples
        --------
        >>> registry = AgentConfigurationRegistry()
        >>> count = registry.load_from_directory("./agents")
        >>> print(f"Loaded {count} configurations")
        """
        dir_path = Path(path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")

        count = 0
        for json_file in sorted(dir_path.glob("*.json")):
            try:
                config = AgentConfiguration.from_json_file(json_file)
                self.register(config)
                count += 1
            except Exception as exc:
                # Log warning but continue processing other files
                import warnings

                warnings.warn(
                    f"Failed to load configuration from {json_file}: {exc}",
                    stacklevel=2,
                )

        return count


# Global default registry instance
_default_registry = AgentConfigurationRegistry()


def get_default_registry() -> AgentConfigurationRegistry:
    """Return the global default registry instance.

    Returns
    -------
    AgentConfigurationRegistry
        Singleton registry for application-wide configuration storage.

    Examples
    --------
    >>> registry = get_default_registry()
    >>> config = AgentConfiguration(name="test", model="gpt-4o-mini")
    >>> registry.register(config)
    """
    return _default_registry


@dataclass(frozen=True, slots=True)
class AgentConfiguration(JSONSerializable):
    """Immutable configuration for building an BaseAgent.

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
    create_agent(run_context_wrapper, prompt_dir, default_model)
        Create an BaseAgent instance from this configuration.
    replace(**changes)
        Create a new AgentConfiguration with specified fields replaced.
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
    >>> config = AgentConfiguration(
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
            raise TypeError("AgentConfiguration.name must be a non-empty str")

        if self.instructions is not None:
            if isinstance(self.instructions, str):
                if not self.instructions.strip():
                    raise ValueError(
                        "AgentConfiguration.instructions must be a non-empty str"
                    )
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

    def create_agent(
        self,
        run_context_wrapper: Any = None,
        prompt_dir: Path | None = None,
        default_model: str | None = None,
    ) -> Any:
        """Create an BaseAgent instance from this configuration.

        This is a convenience method that delegates to BaseAgent.from_configuration().

        Parameters
        ----------
        run_context_wrapper : RunContextWrapper or None, default=None
            Optional wrapper providing runtime context for prompt rendering.
        prompt_dir : Path or None, default=None
            Optional directory holding prompt templates.
        default_model : str or None, default=None
            Optional fallback model identifier if config doesn't specify one.

        Returns
        -------
        BaseAgent
            Configured agent instance ready for execution.

        Examples
        --------
        >>> config = AgentConfiguration(name="helper", model="gpt-4o-mini")
        >>> agent = config.create_agent()
        >>> result = agent.run_sync("Hello!")
        """
        # Import here to avoid circular dependency
        from .base import BaseAgent

        return BaseAgent.from_configuration(
            config=self,
            run_context_wrapper=run_context_wrapper,
            prompt_dir=prompt_dir,
            default_model=default_model,
        )

    def replace(self, **changes: Any) -> AgentConfiguration:
        """Create a new AgentConfiguration with specified fields replaced.

        Since AgentConfiguration is frozen (immutable), this method creates a new
        instance with the specified changes applied. This is useful for
        creating variations of a configuration.

        Parameters
        ----------
        **changes : Any
            Keyword arguments specifying fields to change and their new values.

        Returns
        -------
        AgentConfiguration
            New configuration instance with changes applied.

        Examples
        --------
        >>> config = AgentConfiguration(name="agent1", model="gpt-4o-mini")
        >>> config2 = config.replace(name="agent2", description="Modified")
        >>> config2.name
        'agent2'
        >>> config2.model
        'gpt-4o-mini'
        """
        from dataclasses import replace

        return replace(self, **changes)


__all__ = ["AgentConfiguration", "AgentConfigurationRegistry", "get_default_registry"]
