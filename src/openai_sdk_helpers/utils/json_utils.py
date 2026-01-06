"""JSON serialization helpers for helper types."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from .path_utils import check_filepath

T = TypeVar("T", bound="JSONSerializable")


def _to_jsonable(value: Any) -> Any:
    """Convert common helper types to JSON-serializable forms."""
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
    if hasattr(value, "model_dump"):
        model_dump = getattr(value, "model_dump")
        return model_dump()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, BaseStructure):
        return value.model_dump()
    return value


def coerce_jsonable(value: Any) -> Any:
    """Convert value into a JSON-serializable representation."""
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
    """JSON encoder for common helper types like enums and paths."""

    def default(self, o: Any) -> Any:  # noqa: D401
        """Return JSON-serializable representation of ``o``."""
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
                            # Check if Path is one of the union types
                            if Path in type_args:
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


__all__ = [
    "coerce_jsonable",
    "JSONSerializable",
    "customJSONEncoder",
]
