"""Utilities for resolving instructions from strings or file paths."""

from __future__ import annotations

from pathlib import Path


def resolve_instructions_from_path(instructions: str | Path) -> str:
    """Resolve instructions from a string or file path.

    Parameters
    ----------
    instructions : str or Path
        Either plain-text instructions or a path to a file containing
        instructions.

    Returns
    -------
    str
        The resolved instruction text.

    Raises
    ------
    ValueError
        If instructions is a Path that cannot be read.
    """
    if isinstance(instructions, Path):
        instruction_path = instructions.expanduser()
        try:
            return instruction_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Unable to read instructions at '{instruction_path}': {exc}"
            ) from exc
    return instructions
