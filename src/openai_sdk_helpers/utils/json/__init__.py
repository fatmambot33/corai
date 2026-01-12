"""JSON serialization helpers for dataclasses, Pydantic models, and reference encoding.

This package provides consistent to_json/from_json flows and a JSONEncoder that
handles common types including dataclasses, Pydantic models, and reference encoding.

Package Layout
--------------
utils.py
    to_jsonable(), coerce_jsonable(), customJSONEncoder.
data_class.py
    DataclassJSONSerializable mixin with to_json, to_json_file, from_json, from_json_file.
base_model.py
    BaseModelJSONSerializable for Pydantic, with _serialize_fields/_deserialize_fields hooks.
ref.py
    Reference helpers get_module_qualname, encode_module_qualname, decode_module_qualname.

Public API
----------
to_jsonable(value)
    Convert common types to JSON-safe forms; recursive for containers/dicts.
coerce_jsonable(value)
    Ensures json.dumps succeeds; falls back to str when necessary. Special-cases ResponseBase.
customJSONEncoder
    json.JSONEncoder subclass delegating to to_jsonable.
DataclassJSONSerializable
    Mixin adding to_json(), to_json_file(path) -> str, from_json(data) -> T, from_json_file(path) -> T.
BaseModelJSONSerializable
    Pydantic BaseModel subclass adding to_json() -> dict, to_json_file(path) -> str,
    from_json(data) -> T, from_json_file(path) -> T, plus overridable _serialize_fields(data)
    and _deserialize_fields(data).
get_module_qualname(obj) -> (module, qualname)
    Safe retrieval.
encode_module_qualname(obj) -> dict|None
    {module, qualname} for import reconstruction.
decode_module_qualname(ref) -> object|None
    Import and getattr by encoded reference.
"""

from __future__ import annotations

from .base_model import BaseModelJSONSerializable
from .data_class import DataclassJSONSerializable
from .ref import decode_module_qualname, encode_module_qualname, get_module_qualname
from .utils import coerce_jsonable, customJSONEncoder, to_jsonable

__all__ = [
    "to_jsonable",
    "coerce_jsonable",
    "customJSONEncoder",
    "DataclassJSONSerializable",
    "BaseModelJSONSerializable",
    "get_module_qualname",
    "encode_module_qualname",
    "decode_module_qualname",
]
