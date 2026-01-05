"""Tests for logging configuration module."""

import logging


from openai_sdk_helpers.logging_config import log


class TestLogging:
    """Test simple logging functionality."""

    def test_log_with_default_level(self, caplog) -> None:
        """Should log with default INFO level."""
        caplog.set_level(logging.INFO)
        log("Test message")
        assert "Test message" in caplog.text

    def test_log_with_debug_level(self, caplog) -> None:
        """Should log with DEBUG level when specified."""
        caplog.set_level(logging.DEBUG)
        log("Debug message", level=logging.DEBUG)
        assert "Debug message" in caplog.text

    def test_log_with_custom_logger_name(self, caplog) -> None:
        """Should use custom logger name."""
        caplog.set_level(logging.INFO)
        log("Custom logger", logger_name="custom_logger")
        assert "Custom logger" in caplog.text

    def test_log_does_not_raise(self) -> None:
        """Should not raise exceptions during normal operation."""
        log("Test")  # Should not raise
