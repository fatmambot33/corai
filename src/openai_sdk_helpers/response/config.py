"""Module defining the ResponseConfiguration dataclass for managing OpenAI SDK responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, Sequence, Type, TypeVar
from openai.types.responses.response_text_config_param import ResponseTextConfigParam

from ..structure.base import BaseStructure
from ..response.base import BaseResponse

TIn = TypeVar("TIn", bound="BaseStructure")
TOut = TypeVar("TOut", bound="BaseStructure")


@dataclass(frozen=True, slots=True)
class ResponseConfiguration(Generic[TIn, TOut]):
    """
    Represent an immutable configuration describing input and output structures.

    Encapsulate all metadata required to define how a request is interpreted and
    how a response is structured, while enforcing strict type and runtime safety.

    Parameters
    ----------
    name : str
        Unique configuration identifier. Must be a non-empty string.
    tools : Sequence[object], optional
        Tool definitions associated with the configuration. Default is None.
    schema : ResponseTextConfigParam, optional
        Text response configuration applied at generation time. Default is None.
    input_structure : Type[BaseStructure], optional
        Structure class used to parse or validate input. Must subclass
        BaseStructure. Default is None.
    output_structure : Type[BaseStructure], optional
        Structure class used to format or validate output. Must subclass
        BaseStructure. Default is None.

    Raises
    ------
    TypeError
        If name is not a non-empty string.
        If tools is provided and is not a sequence.
        If input_structure or output_structure is not a class.
        If input_structure or output_structure does not subclass BaseStructure.

    Methods
    -------
    __post_init__()
        Validate configuration invariants and enforce BaseStructure subclassing.

    Examples
    --------
    >>> config = Configuration(
    ...     name="targeting_to_plan",
    ...     tools=None,
    ...     schema=None,
    ...     input_structure=PromptStructure,
    ...     output_structure=WebSearchStructure,
    ... )
    >>> config.name
    'prompt_to_websearch'
    """

    name: str
    tools: Optional[list]
    schema: Optional[ResponseTextConfigParam]
    input_structure: Optional[Type[TIn]]
    output_structure: Optional[Type[TOut]]

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

    def gen_response(self) -> BaseResponse[TOut]:
        """Generate a BaseResponse instance based on the configuration.

        Returns
        -------
        BaseResponse[TOut]
            An instance of BaseResponse configured with the current settings.
        """
        return BaseResponse[TOut](
            instructions="",
            tools=self.tools,
            schema=self.schema,
            output_structure=self.output_structure,
            tool_handlers={},
        )
