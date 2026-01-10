"""Core JSON serialization utilities.

This module provides the core functions for converting common types to
JSON-serializable forms, including to_jsonable, coerce_jsonable, and
customJSONEncoder.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .ref import encode_module_qualname


def to_jsonable(value: Any) -> Any:
    """Convert common types to JSON-safe forms.

    Recursively converts containers, dicts, dataclasses, Pydantic models, enums,
    paths, datetimes, and BaseStructure instances/classes to JSON-serializable forms.
    Private properties (starting with underscore) are excluded from serialization.

    Parameters
    ----------
    value : Any
        Value to convert to JSON-serializable form.

    Returns
    -------
    Any
        JSON-serializable representation of the value.

    Notes
    -----
    Serialization rules:
    - Enums: use enum.value
    - Paths: serialize to string
    - Datetimes: ISO8601 datetime.isoformat()
    - Dataclasses (instances): asdict followed by recursive conversion
    - Pydantic-like objects: use model_dump() if available
    - Dicts/containers: recursively convert values; dict keys coerced to str
    - Private properties: keys starting with underscore are excluded
    - BaseStructure instances: use .model_dump()
    - BaseStructure classes: encode with {module, qualname, "__structure_class__": True}
    - Sets: converted to lists

    Examples
    --------
    >>> from enum import Enum
    >>> class Color(Enum):
    ...     RED = "red"
    >>> to_jsonable(Color.RED)
    'red'
    >>> to_jsonable(Path("/tmp/test"))
    '/tmp/test'
    >>> to_jsonable({"public": 1, "_private": 2})
    {'public': 1}
    """
    return _to_jsonable(value)


def _to_jsonable(value: Any) -> Any:
    """Convert common helper types to JSON-serializable forms (internal)."""
    from openai_sdk_helpers.structure.base import BaseStructure

    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    # Check for BaseStructure class (not instance) before model_dump check
    if isinstance(value, type):
        try:
            if issubclass(value, BaseStructure):
                encoded = encode_module_qualname(value)
                if encoded:
                    encoded["__structure_class__"] = True
                    return encoded
                return str(value)
        except TypeError:
            # Some type-like objects may pass isinstance(value, type) but
            # still not be valid arguments to issubclass; ignore these.
            pass
    if isinstance(value, BaseStructure):
        return value.model_dump()
    # Check for model_dump on instances (after class checks)
    if hasattr(value, "model_dump") and not isinstance(value, type):
        model_dump = getattr(value, "model_dump")
        return model_dump()
    if isinstance(value, dict):
        return {
            str(k): _to_jsonable(v)
            for k, v in value.items()
            if not str(k).startswith("_")
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return value


def coerce_jsonable(value: Any) -> Any:
    """Ensure json.dumps succeeds.

    Falls back to str when necessary. Special-cases BaseResponse.

    Parameters
    ----------
    value : Any
        Value to coerce to JSON-serializable form.

    Returns
    -------
    Any
        JSON-serializable representation, or str(value) as fallback.

    Notes
    -----
    This function first attempts to convert the value using to_jsonable(),
    then validates it can be serialized with json.dumps(). If serialization
    fails, it falls back to str(value).

    Special handling for BaseResponse: serialized as messages.to_json().

    Examples
    --------
    >>> coerce_jsonable({"key": "value"})
    {'key': 'value'}
    >>> class CustomObj:
    ...     def __str__(self):
    ...         return "custom"
    >>> coerce_jsonable(CustomObj())
    'custom'
    """
    from openai_sdk_helpers.response.base import BaseResponse

    if value is None:
        return None
    if isinstance(value, BaseResponse):
        return coerce_jsonable(value.messages.to_json())
    if is_dataclass(value) and not isinstance(value, type):
        return {key: coerce_jsonable(item) for key, item in asdict(value).items()}
    coerced = _to_jsonable(value)
    try:
        json.dumps(coerced)
        return coerced
    except TypeError:
        return str(coerced)


class customJSONEncoder(json.JSONEncoder):
    """JSON encoder delegating to to_jsonable.

    This encoder handles common types like Enum, Path, datetime, dataclasses,
    sets, BaseStructure instances/classes, and Pydantic-like objects.

    Examples
    --------
    >>> import json
    >>> from enum import Enum
    >>> class Color(Enum):
    ...     RED = "red"
    >>> json.dumps({"color": Color.RED}, cls=customJSONEncoder)
    '{"color": "red"}'
    """

    def default(self, o: Any) -> Any:
        """Return JSON-serializable representation of object.

        Parameters
        ----------
        o : Any
            Object to serialize.

        Returns
        -------
        Any
            JSON-serializable representation.
        """
        return _to_jsonable(o)


__all__ = [
    "to_jsonable",
    "coerce_jsonable",
    "customJSONEncoder",
]
