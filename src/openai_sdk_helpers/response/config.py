"""Module defining the ResponseConfiguration dataclass for managing OpenAI SDK responses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, Sequence, Type, TypeVar

from ..config import OpenAISettings
from ..structure.base import BaseStructure
from ..response.base import BaseResponse, ToolHandler
from ..utils.json.data_class import DataclassJSONSerializable
from ..utils.registry import BaseRegistry
from ..utils.instructions import resolve_instructions_from_path

TIn = TypeVar("TIn", bound="BaseStructure")
TOut = TypeVar("TOut", bound="BaseStructure")


class ResponseRegistry(BaseRegistry["ResponseConfiguration"]):
    """Registry for managing ResponseConfiguration instances.

    Inherits from BaseRegistry to provide centralized storage and retrieval
    of response configurations, enabling reusable response specs across the application.

    Examples
    --------
    >>> registry = ResponseRegistry()
    >>> config = ResponseConfiguration(
    ...     name="test",
    ...     instructions="Test instructions",
    ...     tools=None,
    ...     input_structure=None,
    ...     output_structure=None
    ... )
    >>> registry.register(config)
    >>> retrieved = registry.get("test")
    >>> retrieved.name
    'test'
    """

    pass


def get_default_registry() -> ResponseRegistry:
    """Return the global default registry instance.

    Returns
    -------
    ResponseRegistry
        Singleton registry for application-wide configuration storage.

    Examples
    --------
    >>> registry = get_default_registry()
    >>> config = ResponseConfiguration(...)
    >>> registry.register(config)
    """
    return _default_registry


@dataclass(frozen=True, slots=True)
class ResponseConfiguration(DataclassJSONSerializable, Generic[TIn, TOut]):
    """
    Represent an immutable configuration describing input and output structures.

    Encapsulate all metadata required to define how a request is interpreted and
    how a response is structured, while enforcing strict type and runtime safety.
    Inherits from DataclassJSONSerializable to support serialization to JSON format.

    Parameters
    ----------
    name : str
        Unique configuration identifier. Must be a non-empty string.
    instructions : str or Path
        Plain text instructions or a path to a Jinja template file whose
        contents are loaded at runtime.
    tools : Sequence[object], optional
        Tool definitions associated with the configuration. Default is None.
    input_structure : Type[BaseStructure], optional
        Structure class used to parse or validate input. Must subclass
        BaseStructure. Default is None.
    output_structure : Type[BaseStructure], optional
        Structure class used to format or validate output. Schema is
        automatically generated from this structure. Must subclass
        BaseStructure. Default is None.
    system_vector_store : list[str], optional
        Optional list of vector store names to attach as system context.
        Default is None.
    data_path : Path, str, or None, optional
        Optional absolute directory path for storing artifacts. If not provided,
        defaults to get_data_path(class_name). Default is None.

    Raises
    ------
    TypeError
        If name is not a non-empty string.
        If instructions is not a string or Path.
        If tools is provided and is not a sequence.
        If input_structure or output_structure is not a class.
        If input_structure or output_structure does not subclass BaseStructure.
    ValueError
        If instructions is a string that is empty or only whitespace.
    FileNotFoundError
        If instructions is a Path that does not point to a readable file.

    Methods
    -------
    __post_init__()
        Validate configuration invariants and enforce BaseStructure subclassing.
    instructions_text
        Return the resolved instruction content as a string.
    to_json()
        Return a JSON-compatible dict representation (inherited from JSONSerializable).
    to_json_file(filepath)
        Write serialized JSON data to a file path (inherited from JSONSerializable).
    from_json(data)
        Create an instance from a JSON-compatible dict (class method, inherited from JSONSerializable).
    from_json_file(filepath)
        Load an instance from a JSON file (class method, inherited from JSONSerializable).

    Examples
    --------
    >>> config = Configuration(
    ...     name="targeting_to_plan",
    ...     tools=None,
    ...     input_structure=PromptStructure,
    ...     output_structure=WebSearchStructure,
    ... )
    >>> config.name
    'prompt_to_websearch'
    """

    name: str
    instructions: str | Path
    tools: Optional[list]
    input_structure: Optional[Type[TIn]]
    output_structure: Optional[Type[TOut]]
    system_vector_store: Optional[list[str]] = None
    add_output_instructions: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration invariants after initialization.

        Enforce non-empty naming, correct typing of structures, and ensure that
        any declared structure subclasses BaseStructure.

        Raises
        ------
        TypeError
            If name is not a non-empty string.
            If tools is provided and is not a sequence.
            If input_structure or output_structure is not a class.
            If input_structure or output_structure does not subclass BaseStructure.
        """
        if not self.name or not isinstance(self.name, str):
            raise TypeError("Configuration.name must be a non-empty str")

        instructions_value = self.instructions
        if isinstance(instructions_value, str):
            if not instructions_value.strip():
                raise ValueError("Configuration.instructions must be a non-empty str")
        elif isinstance(instructions_value, Path):
            instruction_path = instructions_value.expanduser()
            if not instruction_path.is_file():
                raise FileNotFoundError(
                    f"Instruction template not found: {instruction_path}"
                )
        else:
            raise TypeError("Configuration.instructions must be a str or Path")

        for attr in ("input_structure", "output_structure"):
            cls = getattr(self, attr)
            if cls is None:
                continue
            if not isinstance(cls, type):
                raise TypeError(
                    f"Configuration.{attr} must be a class (Type[BaseStructure]) or None"
                )
            if not issubclass(cls, BaseStructure):
                raise TypeError(f"Configuration.{attr} must subclass BaseStructure")

        if self.tools is not None and not isinstance(self.tools, Sequence):
            raise TypeError("Configuration.tools must be a Sequence or None")

    @property
    def instructions_text(self) -> str:
        """Return the resolved instruction text.

        Returns
        -------
        str
            Plain-text instructions, loading template files when necessary.
        """
        resolved_instructions: str = resolve_instructions_from_path(self.instructions)
        output_instructions = ""
        if self.output_structure is not None and self.add_output_instructions:
            output_instructions = self.output_structure.get_prompt(
                add_enum_values=False
            )
            if output_instructions:
                return f"{resolved_instructions}\n{output_instructions}"

        return resolved_instructions

    def gen_response(
        self,
        *,
        openai_settings: OpenAISettings,
        data_path: Optional[Path] = None,
        tool_handlers: dict[str, ToolHandler] | None = None,
    ) -> BaseResponse[TOut]:
        """Generate a BaseResponse instance based on the configuration.

        Parameters
        ----------
        openai_settings : OpenAISettings
            Authentication and model settings applied to the generated
            :class:`BaseResponse`.
        tool_handlers : dict[str, Callable], optional
            Mapping of tool names to handler callables. Defaults to an empty
            dictionary when not provided.

        Returns
        -------
        BaseResponse[TOut]
            An instance of BaseResponse configured with ``openai_settings``.
        """
        return BaseResponse[TOut](
            name=self.name,
            instructions=self.instructions_text,
            tools=self.tools,
            output_structure=self.output_structure,
            system_vector_store=self.system_vector_store,
            data_path=data_path,
            tool_handlers=tool_handlers,
            openai_settings=openai_settings,
        )


# Global default registry instance
_default_registry = ResponseRegistry()
