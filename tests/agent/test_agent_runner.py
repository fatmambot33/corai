"""Tests for the agent runner convenience functions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_sdk_helpers.agent import runner


@pytest.fixture
def mock_agent():
    """Return a mock agent."""
    return MagicMock()


@patch("openai_sdk_helpers.agent.runner._run_async", new_callable=AsyncMock)
def test_run_async_returns_coroutine(mock_run_async, mock_agent):
    """Test that run_async returns a coroutine."""
    coro = runner.run_async(mock_agent, "test_input")
    assert asyncio.iscoroutine(coro)
    asyncio.run(coro)


@patch("openai_sdk_helpers.agent.runner._run_sync")
def test_run_sync(mock_run_sync, mock_agent):
    """Test the run_sync function."""
    runner.run_sync(mock_agent, "test_input")
    mock_run_sync.assert_called_once_with(
        agent=mock_agent,
        input="test_input",
        context=None,
    )


@patch("openai_sdk_helpers.agent.runner._run_streamed")
def test_run_streamed(mock_run_streamed, mock_agent):
    """Test the run_streamed function."""
    runner.run_streamed(mock_agent, "test_input")
    mock_run_streamed.assert_called_once_with(
        agent=mock_agent,
        input="test_input",
        context=None,
    )
