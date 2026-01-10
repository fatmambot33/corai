"""Reference encoding helpers for object reconstruction.

This module provides helpers for encoding and decoding object references
using module and qualname information, enabling serialization of class
references for later reconstruction.
"""

from __future__ import annotations

import importlib
from typing import Any


def get_module_qualname(obj: Any) -> tuple[str, str] | None:
    """Retrieve module and qualname for an object.

    Safe retrieval that returns None if module or qualname cannot be determined.

    Parameters
    ----------
    obj : Any
        Object to get module and qualname from.

    Returns
    -------
    tuple[str, str] or None
        Tuple of (module, qualname) or None if cannot be determined.

    Examples
    --------
    >>> class MyClass:
    ...     pass
    >>> get_module_qualname(MyClass)
    ('__main__', 'MyClass')
    """
    try:
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        if module and qualname:
            return (module, qualname)
    except Exception:
        pass
    return None


def encode_module_qualname(obj: Any) -> dict[str, Any] | None:
    """Encode object reference for import reconstruction.

    Parameters
    ----------
    obj : Any
        Object to encode (typically a class).

    Returns
    -------
    dict[str, Any] or None
        Dictionary with 'module' and 'qualname' keys, or None if encoding fails.

    Examples
    --------
    >>> class MyClass:
    ...     pass
    >>> encode_module_qualname(MyClass)
    {'module': '__main__', 'qualname': 'MyClass'}
    """
    result = get_module_qualname(obj)
    if result is None:
        return None
    module, qualname = result
    return {"module": module, "qualname": qualname}


def decode_module_qualname(ref: dict[str, Any]) -> Any | None:
    """Import and retrieve object by encoded reference.

    Parameters
    ----------
    ref : dict[str, Any]
        Dictionary with 'module' and 'qualname' keys.

    Returns
    -------
    Any or None
        Retrieved object or None if import/retrieval fails.

    Examples
    --------
    >>> ref = {'module': 'pathlib', 'qualname': 'Path'}
    >>> decode_module_qualname(ref)
    <class 'pathlib.Path'>
    """
    if not isinstance(ref, dict):
        return None

    module_name = ref.get("module")
    qualname = ref.get("qualname")

    if not module_name or not qualname:
        return None

    try:
        module = importlib.import_module(module_name)
        # Handle nested qualnames (e.g., "OuterClass.InnerClass")
        obj = module
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
        return obj
    except (ImportError, AttributeError):
        return None


__all__ = [
    "get_module_qualname",
    "encode_module_qualname",
    "decode_module_qualname",
]
