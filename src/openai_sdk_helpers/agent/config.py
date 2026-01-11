"""Configuration helpers for ``BaseAgent``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents import Agent, Handoff, InputGuardrail, OutputGuardrail, Session
from agents.model_settings import ModelSettings

from ..utils.json.data_class import DataclassJSONSerializable
from ..utils.registry import BaseRegistry
from ..utils.instructions import resolve_instructions_from_path


class AgentConfigurationRegistry(BaseRegistry["AgentConfiguration"]):
    """Registry for managing AgentConfiguration instances.

    Inherits from BaseRegistry to provide centralized storage and retrieval
    of agent configurations, enabling reusable agent specs across the application.

    Examples
    --------
    >>> registry = AgentConfigurationRegistry()
    >>> config = AgentConfiguration(
    ...     name="test_agent",
    ...     model="gpt-4o-mini",
    ...     instructions="Test instructions"
    ... )
    >>> registry.register(config)
    >>> retrieved = registry.get("test_agent")
    >>> retrieved.name
    'test_agent'
    """

    def load_from_directory(
        self,
        path: Path | str,
        *,
        config_class: type["AgentConfiguration"] | None = None,
    ) -> int:
        """Load all agent configurations from JSON files in a directory.

        Parameters
        ----------
        path : Path or str
            Directory path containing JSON configuration files.
        config_class : type[AgentConfiguration], optional
            The configuration class to use for deserialization.
            Defaults to AgentConfiguration.

        Returns
        -------
        int
            Number of configurations successfully loaded and registered.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        NotADirectoryError
            If the path is not a directory.

        Examples
        --------
        >>> registry = AgentConfigurationRegistry()
        >>> count = registry.load_from_directory("./agents")
        >>> print(f"Loaded {count} configurations")
        """
        if config_class is None:
            config_class = AgentConfiguration
        return super().load_from_directory(path, config_class=config_class)


def get_default_registry() -> AgentConfigurationRegistry:
    """Return the global default registry instance.

    Returns
    -------
    AgentConfigurationRegistry
        Singleton registry for application-wide configuration storage.

    Examples
    --------
    >>> registry = get_default_registry()
    >>> config = AgentConfiguration(
    ...     name="test", model="gpt-4o-mini", instructions="Test instructions"
    ... )
    >>> registry.register(config)
    """
    return _default_registry


@dataclass(frozen=True, slots=True)
class AgentConfiguration(DataclassJSONSerializable):
    """Immutable configuration for building a BaseAgent.

    Encapsulates all metadata required to define an agent including its
    instructions, tools, model settings, handoffs, guardrails, and session
    management. Inherits from DataclassJSONSerializable to support serialization.

    This dataclass is frozen (immutable) to ensure thread-safety and
    enable use as dictionary keys. All list-type fields use None as the
    default value rather than mutable defaults like [] to avoid issues
    with shared state across instances.

    Parameters
    ----------
    name : str
        Unique identifier for the agent. Must be a non-empty string.
    instructions : str or Path
        Plain text instructions or a path to a Jinja template file whose
        contents are loaded at runtime. Required field.
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
        Create a BaseAgent instance from this configuration.
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
    instructions: str | Path
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
        are properly formatted.

        Raises
        ------
        TypeError
            If name is not a non-empty string.
            If instructions is not a string or Path.
        ValueError
            If instructions is an empty string.
        FileNotFoundError
            If instructions is a Path that doesn't point to a readable file.
        """
        if not self.name or not isinstance(self.name, str):
            raise TypeError("AgentConfiguration.name must be a non-empty str")

        # Validate instructions (required field, like in Response module)
        instructions_value = self.instructions
        if isinstance(instructions_value, str):
            if not instructions_value.strip():
                raise ValueError(
                    "AgentConfiguration.instructions must be a non-empty str"
                )
        elif isinstance(instructions_value, Path):
            instruction_path = instructions_value.expanduser()
            if not instruction_path.is_file():
                raise FileNotFoundError(
                    f"Instruction template not found: {instruction_path}"
                )
        else:
            raise TypeError("AgentConfiguration.instructions must be a str or Path")

        if self.template_path is not None and isinstance(self.template_path, Path):
            # Validate template_path if it's a Path object
            template = self.template_path.expanduser()
            if not template.exists():
                # We don't raise here because template_path might be relative
                # and resolved later with prompt_dir
                pass

    @property
    def instructions_text(self) -> str:
        """Return the resolved instruction text.

        Returns
        -------
        str
            Plain-text instructions, loading template files when necessary.
        """
        return self._resolve_instructions()

    def _resolve_instructions(self) -> str:
        """Resolve instructions from string or file path."""
        return resolve_instructions_from_path(self.instructions)

    def create_agent(
        self,
        run_context_wrapper: Any = None,
        prompt_dir: Path | None = None,
        default_model: str | None = None,
    ) -> Any:
        """Create a BaseAgent instance from this configuration.

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
        >>> config = AgentConfiguration(
        ...     name="helper", model="gpt-4o-mini", instructions="Help the user"
        ... )
        >>> agent = config.create_agent()
        >>> result = agent.run_sync("Hello!")
        """
        # Import here to avoid circular dependency
        from .base import BaseAgent

        BaseAgent(
            config=self,
            run_context_wrapper=run_context_wrapper,
            prompt_dir=prompt_dir,
            default_model=default_model,
        )
        return BaseAgent(
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
        >>> config = AgentConfiguration(
        ...     name="agent1", model="gpt-4o-mini", instructions="Agent instructions"
        ... )
        >>> config2 = config.replace(name="agent2", description="Modified")
        >>> config2.name
        'agent2'
        >>> config2.model
        'gpt-4o-mini'
        """
        from dataclasses import replace

        return replace(self, **changes)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AgentConfiguration:
        """Create an AgentConfiguration from JSON data.

        Overrides the default JSONSerializable.from_json to properly handle
        the instructions field, converting string paths that look like file
        paths back to Path objects for proper file loading.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing the configuration data.

        Returns
        -------
        AgentConfiguration
            New configuration instance.

        Notes
        -----
        This method attempts to preserve the original type of the instructions
        field. If instructions is a string that represents an existing file path,
        it will be converted to a Path object to ensure proper file loading
        behavior is maintained across JSON round-trips.
        """
        # Make a copy to avoid modifying the input
        data = data.copy()

        # Handle instructions field: if it's a string path to an existing file,
        # convert it back to Path for proper file loading
        if "instructions" in data and data["instructions"] is not None:
            instructions_value = data["instructions"]
            if isinstance(instructions_value, str):
                # Check if it looks like a file path and the file exists
                # This preserves the intended behavior for file-based instructions
                try:
                    potential_path = Path(instructions_value)
                    # Only convert to Path if it's an existing file
                    # This way, plain text instructions stay as strings
                    if potential_path.exists() and potential_path.is_file():
                        data["instructions"] = potential_path
                except (OSError, ValueError):
                    # If path parsing fails, keep it as a string (likely plain text)
                    pass

        # Use the parent class method for the rest
        return super(AgentConfiguration, cls).from_json(data)


# Global default registry instance
_default_registry = AgentConfigurationRegistry()

__all__ = ["AgentConfiguration", "AgentConfigurationRegistry", "get_default_registry"]
