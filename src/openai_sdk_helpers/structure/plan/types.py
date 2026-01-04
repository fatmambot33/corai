"""Type aliases for plan execution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Coroutine, TypeAlias

from .enum import AgentEnum

AgentCallable = Callable[..., object | Coroutine[Any, Any, object]]
AgentRegistry: TypeAlias = (
    Mapping[str, AgentCallable] | Mapping[AgentEnum, AgentCallable]
)

__all__ = ["AgentCallable", "AgentRegistry"]
