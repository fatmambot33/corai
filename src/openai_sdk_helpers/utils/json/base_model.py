"""Pydantic BaseModel JSON serialization support.

This module provides BaseModelJSONSerializable for Pydantic models,
with to_json, to_json_file, from_json, from_json_file methods and
customizable _serialize_fields/_deserialize_fields hooks.
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import inspect
import logging
import ast
from typing import Any, TypeVar, get_args, get_origin
from pydantic import BaseModel
from ...logging import log

from .utils import customJSONEncoder

P = TypeVar("P", bound="BaseModelJSONSerializable")


class BaseModelJSONSerializable(BaseModel):
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
        Customize serialization (override in subclasses).
    _deserialize_fields(data)
        Customize deserialization (override in subclasses).

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
        return self.model_dump()

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
        from .. import check_filepath

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
    def _extract_enum_class(cls, field_type: Any) -> type[Enum] | None:
        """Extract an Enum class from a field's type annotation.

        Handles direct Enum types, list[Enum], and optional Enums.

        Parameters
        ----------
        field_type : Any
            Type annotation of a field.

        Returns
        -------
        type[Enum] or None
            Enum class if found, otherwise None.
        """
        origin = get_origin(field_type)
        args = get_args(field_type)

        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            return field_type
        elif (
            origin is list
            and args
            and inspect.isclass(args[0])
            and issubclass(args[0], Enum)
        ):
            return args[0]
        elif origin is not None:
            # Handle Union types
            for arg in args:
                enum_cls = cls._extract_enum_class(arg)
                if enum_cls:
                    return enum_cls
        return None

    @classmethod
    def _build_enum_field_mapping(cls) -> dict[str, type[Enum]]:
        """Build a mapping from field names to their Enum classes.

        Used by from_raw_input to correctly process enum values from
        raw API responses.

        Returns
        -------
        dict[str, type[Enum]]
            Mapping of field names to Enum types.
        """
        mapping: dict[str, type[Enum]] = {}

        for name, model_field in cls.model_fields.items():
            field_type = model_field.annotation
            enum_cls = cls._extract_enum_class(field_type)

            if enum_cls is not None:
                mapping[name] = enum_cls

        return mapping

    @classmethod
    def from_json(cls: type[P], data: dict[str, Any]) -> P:
        """Construct an instance from a dictionary of raw input data.

        Particularly useful for converting data from OpenAI API tool calls
        or assistant outputs into validated structure instances. Handles
        enum value conversion automatically.

        Parameters
        ----------
        data : dict
            Raw input data dictionary from API response.

        Returns
        -------
        T
            Validated instance of the structure class.

        Examples
        --------
        >>> raw_data = {"title": "Test", "score": 0.95}
        >>> instance = MyStructure.from_raw_input(raw_data)
        """
        mapping = cls._build_enum_field_mapping()
        clean_data = data.copy()

        for field, enum_cls in mapping.items():
            raw_value = clean_data.get(field)

            if raw_value is None:
                continue

            # List of enum values
            if isinstance(raw_value, list):
                converted = []
                for v in raw_value:
                    if isinstance(v, enum_cls):
                        converted.append(v)
                    elif isinstance(v, str):
                        # Check if it's a valid value
                        if v in enum_cls._value2member_map_:
                            converted.append(enum_cls(v))
                        # Check if it's a valid name
                        elif v in enum_cls.__members__:
                            converted.append(enum_cls.__members__[v])
                        else:
                            log(
                                f"[{cls.__name__}] Skipping invalid value for '{field}': '{v}' not in {enum_cls.__name__}",
                                level=logging.WARNING,
                            )
                clean_data[field] = converted

            # Single enum value
            elif (
                isinstance(raw_value, str) and raw_value in enum_cls._value2member_map_
            ):
                clean_data[field] = enum_cls(raw_value)

            elif isinstance(raw_value, enum_cls):
                # already the correct type
                continue

            else:
                log(
                    message=f"[{cls.__name__}] Invalid value for '{field}': '{raw_value}' not in {enum_cls.__name__}",
                    level=logging.WARNING,
                )
                clean_data[field] = None

        return cls(**clean_data)

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
        >>> instance = MyConfig.from_json_file("configuration.json")
        """
        target = Path(filepath)
        if not target.exists():
            raise FileNotFoundError(f"JSON file not found: {target}")

        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return cls.from_json(data)

    @classmethod
    def from_string(cls: type[P], arguments: str) -> P:
        """Parse tool call arguments which may not be valid JSON.

        The OpenAI API is expected to return well-formed JSON for tool arguments,
        but minor formatting issues (such as the use of single quotes) can occur.
        This helper first tries ``json.loads`` and falls back to
        ``ast.literal_eval`` for simple cases.

        Parameters
        ----------
        arguments
            Raw argument string from the tool call.

        Returns
        -------
        dict
            Parsed dictionary of arguments.

        Raises
        ------
        ValueError
            If the arguments cannot be parsed as JSON.

        Examples
        --------
        >>> parse_tool_arguments('{"key": "value"}')["key"]
        'value'
        """
        try:
            structured_data = json.loads(arguments)

        except json.JSONDecodeError:
            try:
                structured_data = ast.literal_eval(arguments)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Invalid JSON arguments: {arguments}. "
                    f"Expected valid JSON or Python literal."
                ) from exc
        return cls.from_json(structured_data)


__all__ = ["BaseModelJSONSerializable"]
