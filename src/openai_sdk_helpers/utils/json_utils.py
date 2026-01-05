"""JSON serialization helpers for helper types."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .path_utils import check_filepath


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
    """Mixin for classes that can be serialized to JSON."""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict representation."""
        if is_dataclass(self) and not isinstance(self, type):
            return {k: _to_jsonable(v) for k, v in asdict(self).items()}
        if hasattr(self, "model_dump"):
            model_dump = getattr(self, "model_dump")
            return _to_jsonable(model_dump())
        return _to_jsonable(self.__dict__)

    def to_json_file(self, filepath: str | Path) -> str:
        """Write serialized JSON data to a file path."""
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


__all__ = [
    "coerce_jsonable",
    "JSONSerializable",
    "customJSONEncoder",
]
