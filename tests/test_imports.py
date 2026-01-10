def test_load_modules_by_path():
    import importlib.util
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_corai = os.path.join(root, "src", "openai_sdk_helpers")

    def _load(name: str, filename: str):
        path = os.path.join(src_corai, filename)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        # Register the module so decorators that rely on ``sys.modules`` work
        # correctly when the module is executed outside the normal import flow.
        import sys

        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("openai_sdk_helpers.environment", "environment.py")
    _load("openai_sdk_helpers.utils.coercion", os.path.join("utils", "coercion.py"))
    _load("openai_sdk_helpers.utils.path_utils", os.path.join("utils", "path_utils.py"))
    _load("openai_sdk_helpers.utils.json", os.path.join("utils", "json", "__init__.py"))
    _load(
        "openai_sdk_helpers.vector_storage.types",
        os.path.join("vector_storage", "types.py"),
    )


def test_top_level_exports():
    import openai_sdk_helpers as sdk

    assert hasattr(sdk, "BaseAgent")
    assert hasattr(sdk, "BaseResponse")
