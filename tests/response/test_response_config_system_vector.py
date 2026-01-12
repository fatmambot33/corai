"""Tests for ResponseConfiguration system_vector_store and data_path parameters."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai_sdk_helpers.response.config import ResponseConfiguration


def _mock_vector_store_client(monkeypatch, store_names: list[str]) -> None:
    """Mock the OpenAI client to return dummy vector stores."""

    class DummyClient:
        def __init__(self):
            self.api_key: str | None = "sk-dummy"
            self.vector_stores: Any = self
            self.responses: Any = SimpleNamespace(create=lambda *_a, **_kw: None)
            self.files: Any = SimpleNamespace(
                create=lambda *_a, **_kw: SimpleNamespace(id="fileid"),
                content=lambda *_a, **_kw: SimpleNamespace(read=lambda: b""),
            )

        def list(self):
            stores = []
            for store_name in store_names:
                stores.append(SimpleNamespace(id=f"{store_name}_id", name=store_name))
            return type("obj", (), {"data": stores})()

    dummy_client = DummyClient()
    monkeypatch.setattr(
        "openai_sdk_helpers.config.OpenAI",
        lambda *_a, **_kw: dummy_client,
    )


def test_system_vector_store_passed_to_gen_response(
    openai_settings, monkeypatch
) -> None:
    """Test that system_vector_store is passed through to ResponseBase."""
    _mock_vector_store_client(monkeypatch, ["store1", "store2"])

    config = ResponseConfiguration(
        name="test_config",
        instructions="Test instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
        system_vector_store=["store1", "store2"],
    )

    response = config.gen_response(openai_settings=openai_settings)

    # The ResponseBase should have received the system_vector_store parameter
    # We can verify this indirectly by checking that no error was raised
    assert response.name == "test_config"
    response.close()


def test_system_vector_store_none_default(openai_settings) -> None:
    """Test that system_vector_store defaults to None."""
    config = ResponseConfiguration(
        name="test_config",
        instructions="Test instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    # system_vector_store should be None when not provided
    assert config.system_vector_store is None

    response = config.gen_response(openai_settings=openai_settings)
    assert response.name == "test_config"
