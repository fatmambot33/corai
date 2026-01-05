"""Tests for ResponseConfiguration instruction handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import Field

from openai_sdk_helpers.response.config import ResponseConfiguration
from openai_sdk_helpers.structure.base import BaseStructure


def _build_config(instructions: str | Path) -> ResponseConfiguration:
    return ResponseConfiguration(
        name="unit",
        instructions=instructions,
        tools=None,
        input_structure=None,
        output_structure=None,
    )


def test_instructions_text_returns_plain_string() -> None:
    config = _build_config("Use direct instructions.")
    assert config.instructions_text == "Use direct instructions."


def test_instructions_text_reads_template_file(tmp_path: Path) -> None:
    template_path = tmp_path / "template.jinja"
    template_path.write_text("Template instructions", encoding="utf-8")

    config = _build_config(template_path)
    assert config.instructions_text == "Template instructions"


def test_empty_string_instructions_raise_value_error() -> None:
    with pytest.raises(ValueError):
        _build_config("   ")


def test_missing_template_raises_file_not_found(tmp_path: Path) -> None:
    missing_template = tmp_path / "missing.jinja"
    with pytest.raises(FileNotFoundError):
        _build_config(missing_template)


def test_invalid_instruction_type_raises_type_error() -> None:
    invalid_instructions = cast(Any, 123)
    with pytest.raises(TypeError):
        ResponseConfiguration(
            name="unit",
            instructions=invalid_instructions,
            tools=None,
            input_structure=None,
            output_structure=None,
        )


class _SampleOutput(BaseStructure):
    """Sample output structure for instruction generation tests."""

    summary: str = Field(description="Brief summary of the content")


def test_output_instructions_are_appended(openai_settings) -> None:
    config = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=_SampleOutput,
    )

    response = config.gen_response(openai_settings=openai_settings)

    expected_output = _SampleOutput.get_prompt(add_enum_values=False)
    expected_instructions = f"{config.instructions_text}\n{expected_output}"

    assert response._instructions == expected_instructions


def test_output_instructions_can_be_skipped(openai_settings) -> None:
    config = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=_SampleOutput,
    )

    response = config.gen_response(
        openai_settings=openai_settings, add_output_instructions=False
    )

    expected_instructions = config.instructions_text

    assert response._instructions == expected_instructions


def test_no_output_structure_ignores_add_output_instructions(
    openai_settings,
) -> None:
    """Test that when output_structure is None, add_output_instructions has no effect."""
    config = ResponseConfiguration(
        name="unit",
        instructions="Base instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    # Test with add_output_instructions=True (default)
    response_with_flag = config.gen_response(
        openai_settings=openai_settings, add_output_instructions=True
    )

    # Test with add_output_instructions=False
    response_without_flag = config.gen_response(
        openai_settings=openai_settings, add_output_instructions=False
    )

    # Both should produce the same result: just the base instructions
    assert response_with_flag._instructions == config.instructions_text
    assert response_without_flag._instructions == config.instructions_text
