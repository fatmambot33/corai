import importlib.util
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_OPENAI_SDK_HELPERS = os.path.join(ROOT, "src", "openai_sdk_helpers")


def _load_module(name: str, filename: str):
    path = os.path.join(SRC_OPENAI_SDK_HELPERS, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # register module name so decorators (dataclass) can find module
    import sys

    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


types_mod = _load_module(
    "openai_sdk_helpers.vector_storage.types",
    os.path.join("vector_storage", "types.py"),
)
VectorStorageFileInfo = types_mod.VectorStorageFileInfo
VectorStorageFileStats = types_mod.VectorStorageFileStats


def test_vector_storage_dataclasses_basic():
    info = VectorStorageFileInfo(name="f.txt", id="1", status="success")
    assert info.name == "f.txt"

    stats = VectorStorageFileStats(total=1, success=1, fail=0, errors=[info])
    assert stats.total == 1
    assert stats.errors[0].id == "1"
