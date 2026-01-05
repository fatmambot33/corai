"""Type coercion and collection normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


def coerce_optional_float(value: object) -> float | None:
    """Return a float when the provided value can be coerced, otherwise None.

    Handles float, int, and string inputs. Empty strings or None return None.

    Parameters
    ----------
    value : object
        Value to convert into a float. Strings must be parseable as floats.

    Returns
    -------
    float or None
        Converted float value or None if the input is None.

    Raises
    ------
    ValueError
        If a non-empty string cannot be converted to a float.
    TypeError
        If the value is not a float-compatible type.
    """
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("timeout must be a float-compatible value") from exc
    raise TypeError("timeout must be a float, int, str, or None")


def coerce_optional_int(value: object) -> int | None:
    """Return an int when the provided value can be coerced, otherwise None.

    Handles int, float (if whole number), and string inputs. Empty strings
    or None return None. Booleans are not considered valid integers.

    Parameters
    ----------
    value : object
        Value to convert into an int. Strings must be parseable as integers.

    Returns
    -------
    int or None
        Converted integer value or None if the input is None.

    Raises
    ------
    ValueError
        If a non-empty string cannot be converted to an integer.
    TypeError
        If the value is not an int-compatible type.
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError("max_retries must be an int-compatible value") from exc
    raise TypeError("max_retries must be an int, str, or None")


def coerce_dict(value: object) -> dict[str, Any]:
    """Return a string-keyed dictionary built from value if possible.

    Converts Mapping objects to dictionaries. None returns an empty dict.

    Parameters
    ----------
    value : object
        Mapping-like value to convert. None yields an empty dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of value.

    Raises
    ------
    TypeError
        If the value cannot be treated as a mapping.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("extra_client_kwargs must be a mapping or None")


def ensure_list(value: Iterable[T] | T | None) -> list[T]:
    """Normalize a single item or iterable into a list.

    Converts None to empty list, tuples to lists, and wraps single items in a list.

    Parameters
    ----------
    value : Iterable[T] | T | None
        Item or iterable to wrap. None yields an empty list.

    Returns
    -------
    list[T]
        Normalized list representation of value.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]  # type: ignore[list-item]


__all__ = [
    "coerce_optional_float",
    "coerce_optional_int",
    "coerce_dict",
    "ensure_list",
]
