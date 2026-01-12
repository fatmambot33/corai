"""Configuration helpers for ``AgentBase``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

from agents import Agent, Handoff, InputGuardrail, OutputGuardrail, Session
from agents.model_settings import ModelSettings

from ..utils.json.data_class import DataclassJSONSerializable
from ..utils.registry import RegistryBase
from ..utils.instructions import resolve_instructions_from_path
from ..structure.base import StructureBase


class AgentRegistry(RegistryBase["AgentConfiguration"]):
    """Registry for managing AgentConfiguration instances.

    Inherits from RegistryBase to provide centralized storage and retrieval
    of agent configurations, enabling reusable agent specs across the application.

    Examples
    --------
    >>> registry = AgentRegistry()
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
        >>> registry = AgentRegistry()
        >>> count = registry.load_from_directory("./agents")
        >>> print(f"Loaded {count} configurations")
        """
        if config_class is None:
            config_class = AgentConfiguration
        return super().load_from_directory(path, config_class=config_class)


def get_default_registry() -> AgentRegistry:
    """Return the global default registry instance.

    Returns
    -------
    AgentRegistry
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
    """Immutable configuration for building a AgentBase.

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
    input_structure : type[StructureBase], optional
        Structure class describing the agent input. Default is None.
    output_structure : type[StructureBase], optional
        Structure class describing the agent output. Default is None.
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
    resolve_prompt_path(prompt_dir)
        Resolve the prompt template path for this configuration.
    gen_agent(run_context_wrapper, prompt_dir, default_model)
        Create a AgentBase instance from this configuration.
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
    input_structure: Optional[Type[StructureBase]] = None
    output_structure: Optional[Type[StructureBase]] = None
    tools: Optional[list] = None
    model_settings: Optional[ModelSettings] = None
    handoffs: Optional[list[Agent | Handoff]] = None
    input_guardrails: Optional[list[InputGuardrail]] = None
    output_guardrails: Optional[list[OutputGuardrail]] = None
    session: Optional[Session] = None
    _instructions_cache: Optional[str] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate configuration invariants after initialization.

        Ensures that the name is a non-empty string and that instructions
        are properly formatted.

        Raises
        ------
        TypeError
            If name is not a non-empty string.
            If instructions is not a string or Path.
            If input_structure or output_structure is not a class.
            If input_structure or output_structure does not subclass StructureBase.
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

        for attr in ("input_structure", "output_structure"):
            cls = getattr(self, attr)
            if cls is None:
                continue
            if not isinstance(cls, type):
                raise TypeError(
                    f"AgentConfiguration.{attr} must be a class (Type[StructureBase]) or None"
                )
            if not issubclass(cls, StructureBase):
                raise TypeError(
                    f"AgentConfiguration.{attr} must subclass StructureBase"
                )

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
        cached = self._instructions_cache
        if cached is None:
            cached = self._resolve_instructions()
            object.__setattr__(self, "_instructions_cache", cached)
        return cached

    def _resolve_instructions(self) -> str:
        """Resolve instructions from string or file path."""
        return resolve_instructions_from_path(self.instructions)

    def resolve_prompt_path(self, prompt_dir: Path | None = None) -> Path | None:
        """Resolve the prompt template path for this configuration.

        Parameters
        ----------
        prompt_dir : Path or None, default=None
            Directory holding prompt templates when a relative path is needed.

        Returns
        -------
        Path or None
            Resolved prompt path if a template is configured or discovered.
        """
        if self.template_path:
            return Path(self.template_path)
        if prompt_dir is not None:
            return prompt_dir / f"{self.name}.jinja"
        return None

    def gen_agent(
        self,
        run_context_wrapper: Any = None,
        prompt_dir: Path | None = None,
        default_model: str | None = None,
    ) -> Any:
        """Create a AgentBase instance from this configuration.

        This is a convenience method that instantiates ``AgentBase`` directly.

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
        AgentBase
            Configured agent instance ready for execution.

        Examples
        --------
        >>> config = AgentConfiguration(
        ...     name="helper", model="gpt-4o-mini", instructions="Help the user"
        ... )
        >>> agent = config.gen_agent()
        >>> result = agent.run_sync("Hello!")
        """
        # Import here to avoid circular dependency
        from .base import AgentBase

        return AgentBase(
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

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict representation.

        Returns
        -------
        dict[str, Any]
            Serialized configuration data without cached fields.
        """
        data = DataclassJSONSerializable.to_json(self)
        data.pop("_instructions_cache", None)
        return data

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
        data.pop("_instructions_cache", None)

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
_default_registry = AgentRegistry()

__all__ = ["AgentConfiguration", "AgentRegistry", "get_default_registry"]
