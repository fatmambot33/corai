from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent.base import AgentBase
from openai_sdk_helpers.agent.configuration import AgentConfiguration
from openai_sdk_helpers.structure import SummaryStructure
from openai_sdk_helpers.response.base import ResponseBase


class _StubAgentBase(AgentBase):
    """Minimal AgentBase subclass for testing async aliases.

    Note: Intentionally bypasses parent __init__ to set up minimal test fixture
    without requiring full agent initialization (templates, configuration validation, etc).
    """

    def __init__(self) -> None:
        # Bypass the parent initializer by setting the required attributes.
        # This is intentional for testing - we don't want full agent setup.
        configuration = AgentConfiguration(
            name="stub",
            instructions="Test instructions",
            model="model",
            description="",
            tools=None,
            output_structure=SummaryStructure,
            template_path=None,
        )
        self._configuration = configuration
        self._output_structure = SummaryStructure
        self._run_context_wrapper = None
        self._handoffs = None
        self._input_guardrails = None
        self._output_guardrails = None
        self._session = None
        self._template = MagicMock(render=MagicMock(return_value=""))

    def get_agent(self) -> Any:  # pragma: no cover - mocked in tests
        return MagicMock()


class _StubResponseBase(ResponseBase[Any]):
    """Minimal ResponseBase subclass for alias testing."""

    def __init__(self) -> None:
        # Avoid base initialization; only attributes used in tests are set.
        self._model = "model"
        self._output_structure = None
        self.messages = MagicMock()


@patch("openai_sdk_helpers.agent.base.run_streamed")
@patch("openai_sdk_helpers.agent.base.run_async", new_callable=AsyncMock)
@patch("openai_sdk_helpers.agent.base.run_sync")
def test_agent_base_run_aliases(
    mock_run_agent_sync: MagicMock,
    mock_run_agent: AsyncMock,
    mock_run_agent_streamed: MagicMock,
) -> None:
    """Ensure AgentBase convenience helpers call the private runners directly."""

    mock_run_agent.return_value = "async-result"
    mock_run_agent_sync.return_value = "sync-result"
    mock_run_agent_streamed.return_value = MagicMock(
        final_output_as=lambda *_: "stream-result"
    )
    agent = _StubAgentBase()

    result_run = agent.run_sync(input="hello")
    result_run_async = asyncio.run(agent.run_async(input="hello"))
    result_stream = agent.run_streamed(input="hello", output_structure=SummaryStructure)

    assert result_run == "sync-result"
    assert result_run_async == "async-result"
    assert result_stream == "stream-result"
    mock_run_agent_sync.assert_called_once()
    assert mock_run_agent.await_count == 1
    mock_run_agent_streamed.assert_called_once()


@patch.object(_StubResponseBase, "run_sync", return_value="sync-result")
@patch.object(_StubResponseBase, "run_async", new_callable=AsyncMock)
def test_response_base_run_aliases(
    mock_run_response_async: AsyncMock, mock_run_response: MagicMock
) -> None:
    """Validate ResponseBase exposes run, run_async, and run_streamed aliases."""

    mock_run_response_async.return_value = "async-result"
    response = _StubResponseBase()

    assert response.run_sync(content="hello") == "sync-result"
    assert asyncio.run(response.run_async(content="hello")) == "async-result"
    assert response.run_streamed(content="hello") == "async-result"
    mock_run_response.assert_called_once_with(content="hello")
    assert mock_run_response_async.await_count == 2
