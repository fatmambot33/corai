"""JSON serialization helpers for helper types.

This module provides consistent to_json/from_json flows and a JSONEncoder that
handles common types including dataclasses, Pydantic models, and reference encoding.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from .path_utils import check_filepath

T = TypeVar("T", bound="JSONSerializable")
P = TypeVar("P", bound="BaseModelJSONSerializable")


# Reference encoding helpers


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


# Serialization functions


def to_jsonable(value: Any) -> Any:
    """Convert common types to JSON-safe forms.

    Recursively converts containers, dicts, dataclasses, Pydantic models, enums,
    paths, datetimes, and BaseStructure instances/classes to JSON-serializable forms.

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
    if isinstance(value, type) and issubclass(value, BaseStructure):
        encoded = encode_module_qualname(value)
        if encoded:
            encoded["__structure_class__"] = True
            return encoded
        return str(value)
    if isinstance(value, BaseStructure):
        return value.model_dump()
    # Check for model_dump on instances (after class checks)
    if hasattr(value, "model_dump") and not isinstance(value, type):
        model_dump = getattr(value, "model_dump")
        return model_dump()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
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


class JSONSerializable:
    """Mixin for classes that can be serialized to and from JSON.

    Methods
    -------
    to_json()
        Return a JSON-compatible dict representation.
    to_json_file(filepath)
        Write serialized JSON data to a file path.
    from_json(data)
        Create an instance from a JSON-compatible dict (class method).
    from_json_file(filepath)
        Load an instance from a JSON file (class method).
    """

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict representation."""
        if is_dataclass(self) and not isinstance(self, type):
            return {k: _to_jsonable(v) for k, v in asdict(self).items()}
        if hasattr(self, "model_dump"):
            model_dump = getattr(self, "model_dump")
            return _to_jsonable(model_dump())
        return _to_jsonable(self.__dict__)

    def to_json_file(self, filepath: str | Path) -> str:
        """Write serialized JSON data to a file path.

        Parameters
        ----------
        filepath : str or Path
            Path where the JSON file will be written.

        Returns
        -------
        str
            Absolute path to the written file.
        """
        target = Path(filepath)
        check_filepath(fullfilepath=str(target))
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(
                self.to_json(),
                handle,
                indent=2,
                ensure_ascii=False,
                cls=customJSONEncoder,
            )
        return str(target)

    @classmethod
    def from_json(cls: type[T], data: dict[str, Any]) -> T:
        """Create an instance from a JSON-compatible dict.

        For dataclasses, this reconstructs Path objects and passes the
        dict keys directly as constructor arguments.

        Parameters
        ----------
        data : dict[str, Any]
            JSON-compatible dictionary containing the instance data.

        Returns
        -------
        T
            New instance of the class.

        Examples
        --------
        >>> json_data = {"name": "test", "path": "/tmp/data"}
        >>> instance = MyClass.from_json(json_data)
        """
        if is_dataclass(cls):
            # Get resolved field types using get_type_hints
            try:
                field_types = get_type_hints(cls)
            except Exception:
                # Fallback to raw annotations if get_type_hints fails
                field_types = {f.name: f.type for f in fields(cls)}

            converted_data = {}

            for key, value in data.items():
                if key in field_types:
                    field_type = field_types[key]

                    # Check if this field should be converted to Path
                    should_convert_to_path = False

                    if field_type is Path:
                        should_convert_to_path = True
                    else:
                        # Handle Union/Optional types
                        origin = get_origin(field_type)
                        if origin is Union:
                            type_args = get_args(field_type)
                            # Only convert to Path if:
                            # 1. Path is in the union AND
                            # 2. str is NOT in the union (to avoid converting string fields)
                            #    OR the field name suggests it's a path (contains "path")
                            if Path in type_args:
                                if str not in type_args:
                                    # Path-only union (e.g., Union[Path, None])
                                    should_convert_to_path = True
                                elif "path" in key.lower():
                                    # Field name contains "path", likely meant to be a path
                                    should_convert_to_path = True

                    # Convert string to Path if needed
                    if (
                        should_convert_to_path
                        and value is not None
                        and isinstance(value, str)
                    ):
                        converted_data[key] = Path(value)
                    else:
                        converted_data[key] = value
                else:
                    converted_data[key] = value

            return cls(**converted_data)  # type: ignore[return-value]

        # For non-dataclass types, try to instantiate with data as kwargs
        return cls(**data)  # type: ignore[return-value]

    @classmethod
    def from_json_file(cls: type[T], filepath: str | Path) -> T:
        """Load an instance from a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file to load.

        Returns
        -------
        T
            New instance of the class loaded from the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        >>> instance = MyClass.from_json_file("config.json")
        """
        target = Path(filepath)
        if not target.exists():
            raise FileNotFoundError(f"JSON file not found: {target}")

        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return cls.from_json(data)


# Type aliases and subclasses

# Alias for dataclass usage
DataclassJSONSerializable = JSONSerializable


class BaseModelJSONSerializable:
    """Pydantic BaseModel subclass with JSON serialization support.

    Adds to_json(), to_json_file(path), from_json(data), from_json_file(path),
    plus overridable _serialize_fields(data) and _deserialize_fields(data) hooks.

    Methods
    -------
    to_json()
        Return a JSON-compatible dict representation.
    to_json_file(filepath)
        Write serialized JSON data to a file path.
    from_json(data)
        Create an instance from a JSON-compatible dict (class method).
    from_json_file(filepath)
        Load an instance from a JSON file (class method).
    _serialize_fields(data)
        Hook to customize serialization (override in subclasses).
    _deserialize_fields(data)
        Hook to customize deserialization (override in subclasses).

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class MyConfig(BaseModelJSONSerializable, BaseModel):
    ...     name: str
    ...     value: int
    >>> cfg = MyConfig(name="test", value=42)
    >>> cfg.to_json()
    {'name': 'test', 'value': 42}
    """

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict representation.

        Returns
        -------
        dict[str, Any]
            Serialized model data.
        """
        if hasattr(self, "model_dump"):
            data = getattr(self, "model_dump")()
        else:
            data = self.__dict__.copy()
        return self._serialize_fields(_to_jsonable(data))

    def to_json_file(self, filepath: str | Path) -> str:
        """Write serialized JSON data to a file path.

        Parameters
        ----------
        filepath : str or Path
            Path where the JSON file will be written.

        Returns
        -------
        str
            Absolute path to the written file.
        """
        target = Path(filepath)
        check_filepath(fullfilepath=str(target))
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(
                self.to_json(),
                handle,
                indent=2,
                ensure_ascii=False,
                cls=customJSONEncoder,
            )
        return str(target)

    def _serialize_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Customize field serialization.

        Override this method in subclasses to add custom serialization logic.

        Parameters
        ----------
        data : dict[str, Any]
            Pre-serialized data dictionary.

        Returns
        -------
        dict[str, Any]
            Modified data dictionary.
        """
        return data

    @classmethod
    def _deserialize_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Customize field deserialization.

        Override this method in subclasses to add custom deserialization logic.

        Parameters
        ----------
        data : dict[str, Any]
            Raw data dictionary from JSON.

        Returns
        -------
        dict[str, Any]
            Modified data dictionary.
        """
        return data

    @classmethod
    def from_json(cls: type[P], data: dict[str, Any]) -> P:
        """Create an instance from a JSON-compatible dict.

        Parameters
        ----------
        data : dict[str, Any]
            JSON-compatible dictionary containing the instance data.

        Returns
        -------
        P
            New instance of the class.

        Examples
        --------
        >>> json_data = {"name": "test", "value": 42}
        >>> instance = MyConfig.from_json(json_data)
        """
        processed_data = cls._deserialize_fields(data)
        return cls(**processed_data)  # type: ignore[return-value]

    @classmethod
    def from_json_file(cls: type[P], filepath: str | Path) -> P:
        """Load an instance from a JSON file.

        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file to load.

        Returns
        -------
        P
            New instance of the class loaded from the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        >>> instance = MyConfig.from_json_file("config.json")
        """
        target = Path(filepath)
        if not target.exists():
            raise FileNotFoundError(f"JSON file not found: {target}")

        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return cls.from_json(data)


__all__ = [
    "to_jsonable",
    "coerce_jsonable",
    "customJSONEncoder",
    "JSONSerializable",
    "DataclassJSONSerializable",
    "BaseModelJSONSerializable",
    "get_module_qualname",
    "encode_module_qualname",
    "decode_module_qualname",
]
