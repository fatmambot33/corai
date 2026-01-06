"""Configuration helpers for ``AgentBase``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents import Agent, Handoff, InputGuardrail, OutputGuardrail, Session
from agents.model_settings import ModelSettings

from ..utils import JSONSerializable


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


__all__ = ["AgentConfig"]
