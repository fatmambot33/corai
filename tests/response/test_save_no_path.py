from pathlib import Path

from openai_sdk_helpers import environment
from openai_sdk_helpers.response.base import BaseResponse


def test_save_defaults_to_data_path(monkeypatch, tmp_path, openai_settings):
    """Test that save() writes to the default data path when no filepath is provided."""
    data_root = tmp_path / "data"

    def fake_get_data_path(name: str) -> Path:
        return data_root / name

    monkeypatch.setattr(environment, "get_data_path", fake_get_data_path)
    r = BaseResponse(
        name="test",
        instructions="hi",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    r.save()

    expected_path = data_root / "BaseResponse" / "test" / f"{str(r.uuid).lower()}.json"
    assert expected_path.exists()
