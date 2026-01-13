"""Centralized logging configuration for openai-sdk-helpers."""

import logging


def log(
    message: str,
    level: int = logging.INFO,
    *,
    logger_name: str = "openai_sdk_helpers",
) -> None:
    """Log a message using Python's standard logging.

    Parameters
    ----------
    message : str
        The message to log.
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
        Default is logging.INFO.
    logger_name : str
        Name of the logger. Default is "openai_sdk_helpers".

    Examples
    --------
    >>> from openai_sdk_helpers.logging_config import log
    >>> log("Operation completed")
    >>> log("Debug info", level=logging.DEBUG)
    """
    logger = logging.getLogger(logger_name)
    logger.log(level, message)


__all__ = ["log"]
