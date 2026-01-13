"""Dataclass JSON serialization mixin.

This module provides the DataclassJSONSerializable mixin for dataclasses,
adding to_json, to_json_file, from_json, and from_json_file methods.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from ..path_utils import check_filepath
from .utils import _to_jsonable, customJSONEncoder

T = TypeVar("T", bound="DataclassJSONSerializable")


class DataclassJSONSerializable:
    """Mixin for dataclasses that can be serialized to and from JSON.

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

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from pathlib import Path
    >>> @dataclass
    ... class MyData(DataclassJSONSerializable):
    ...     name: str
    ...     path: Path
    >>> instance = MyData(name="test", path=Path("/tmp/data"))
    >>> json_data = instance.to_json()
    >>> restored = MyData.from_json(json_data)
    """

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict representation.

        Returns
        -------
        dict[str, Any]
            Serialized data dictionary.
        """
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
        >>> instance = MyClass.from_json_file("configuration.json")
        """
        target = Path(filepath)
        if not target.exists():
            raise FileNotFoundError(f"JSON file not found: {target}")

        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return cls.from_json(data)


__all__ = ["DataclassJSONSerializable"]
