"""Pydantic BaseModel JSON serialization support.

This module provides BaseModelJSONSerializable for Pydantic models,
with to_json, to_json_file, from_json, from_json_file methods and
customizable _serialize_fields/_deserialize_fields hooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar
from pydantic import BaseModel
from ..path_utils import check_filepath
from .utils import _to_jsonable, customJSONEncoder

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


__all__ = ["BaseModelJSONSerializable"]
