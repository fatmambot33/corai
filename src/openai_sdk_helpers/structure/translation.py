"""Structured output model for translations."""

from __future__ import annotations

from .base import StructureBase, spec_field


class TranslationStructure(StructureBase):
    """Structured representation of translated text.

    Attributes
    ----------
    text : str
        Translated text output from the agent.

    Methods
    -------
    print()
        Return the formatted model fields.

    Examples
    --------
    >>> translation = TranslationStructure(text="Hola mundo")
    >>> print(translation.text)
    'Hola mundo'
    """

    text: str = spec_field(
        "text",
        description="Translated text output from the agent.",
        examples=["Hola mundo", "Bonjour le monde"],
    )
