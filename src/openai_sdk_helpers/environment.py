"""Environment helpers for openai-sdk-helpers.

This module provides utility functions and constants for managing paths
and environment-specific configuration.

Constants
---------
DATETIME_FMT : str
    Standard datetime format string for file naming ("%Y%m%d_%H%M%S").
DEFAULT_MODEL : str
    Default OpenAI model identifier ("gpt-4o-mini").

Functions
---------
get_data_path(name)
    Return a writable data directory for the given module name.
"""

from __future__ import annotations

from pathlib import Path

from openai_sdk_helpers.utils import ensure_directory

DATETIME_FMT = "%Y%m%d_%H%M%S"
DEFAULT_MODEL = "gpt-4o-mini"


def get_data_path(name: str) -> Path:
    """Return a writable data directory for the given module name.

    Creates a module-specific directory under ~/.openai-sdk-helpers/ for
    storing data, logs, or other persistent files.

    Parameters
    ----------
    name : str
        Name of the module requesting a data directory.

    Returns
    -------
    Path
        Directory path under ~/openai-sdk-helpers specific to name.
        The directory is created if it does not exist.

    Examples
    --------
    >>> from openai_sdk_helpers.environment import get_data_path
    >>> path = get_data_path("my_module")
    >>> path.exists()
    True
    """
    # Use the workspace 'data' directory, with a subdirectory for the module name
    base = Path(__file__).parent.parent.parent / "data"
    path = base / name
    return ensure_directory(path)
