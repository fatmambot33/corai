"""Streamlit application utilities for configuration-driven chat interfaces.

This module provides configuration management and loading utilities for building
Streamlit-based chat applications powered by OpenAI response handlers. It enables
rapid deployment of conversational AI interfaces with minimal boilerplate.

Classes
-------
StreamlitAppConfig
    Validated configuration for Streamlit chat applications.
StreamlitAppRegistry
    Registry for storing Streamlit app configurations.

Functions
---------
_load_configuration
    Load configuration with user-friendly error handling for Streamlit UI.
"""

from .config import (
    StreamlitAppConfig,
    StreamlitAppRegistry,
    _load_configuration,
)

__all__ = [
    "StreamlitAppConfig",
    "StreamlitAppRegistry",
    "_load_configuration",
]
