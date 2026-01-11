"""File and path utilities."""

from __future__ import annotations

from pathlib import Path


def check_filepath(
    filepath: Path | None = None, *, fullfilepath: str | None = None
) -> Path:
    """Ensure the parent directory for a file path exists.

    Creates parent directories as needed. Exactly one of ``filepath`` or
    ``fullfilepath`` must be provided.

    Parameters
    ----------
    filepath : Path or None, optional
        Path object to validate. Mutually exclusive with ``fullfilepath``.
    fullfilepath : str or None, optional
        String path to validate. Mutually exclusive with ``filepath``.

    Returns
    -------
    Path
        Path object representing the validated file path.

    Raises
    ------
    ValueError
        If both or neither of filepath and fullfilepath are provided.
    IOError
        If the directory cannot be created.

    Examples
    --------
    >>> check_filepath(fullfilepath="/tmp/my_app/data.json")
    Path('/tmp/my_app/data.json')
    """
    if filepath is None and fullfilepath is None:
        raise ValueError("filepath or fullfilepath is required.")
    if fullfilepath is not None:
        target = Path(fullfilepath)
    elif filepath is not None:
        target = Path(filepath)
    else:
        raise ValueError("filepath or fullfilepath is required.")
    ensure_directory(target.parent)
    return target


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists and return it.

    Parameters
    ----------
    path : Path
        The directory path to check and create if necessary.

    Returns
    -------
    Path
        The validated Path object for the directory.

    Raises
    ------
    IOError
        If the directory cannot be created.

    Examples
    --------
    >>> ensure_directory(Path("/tmp/my_app"))
    Path('/tmp/my_app')
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["check_filepath", "ensure_directory"]
