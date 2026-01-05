"""Deprecation utilities for managing deprecated features.

This module provides infrastructure for marking and managing deprecated
functions, classes, and features with consistent warning messages.

Functions
---------
deprecated
    Decorator to mark functions or classes as deprecated.
warn_deprecated
    Emit a deprecation warning with optional custom message.

Classes
-------
DeprecationHelper
    Utility class for managing deprecation warnings and versions.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class DeprecationHelper:
    """Utility class for managing deprecation warnings.

    Provides consistent formatting and control of deprecation warnings
    across the package.

    Methods
    -------
    warn
        Emit a deprecation warning with standard formatting.
    """

    @staticmethod
    def warn(
        feature_name: str,
        removal_version: str,
        alternative: str | None = None,
        extra_message: str | None = None,
    ) -> None:
        """Emit a deprecation warning for a feature.

        Parameters
        ----------
        feature_name : str
            Name of the deprecated feature (e.g., "MyClass.old_method").
        removal_version : str
            Version in which the feature will be removed.
        alternative : str, optional
            Recommended alternative to use instead.
        extra_message : str, optional
            Additional context or migration instructions.

        Raises
        ------
        DeprecationWarning
            Always issues a DeprecationWarning to stderr.
        """
        msg = f"{feature_name} is deprecated and will be removed in version {removal_version}."
        if alternative:
            msg += f" Use {alternative} instead."
        if extra_message:
            msg += f" {extra_message}"

        warnings.warn(msg, DeprecationWarning, stacklevel=3)


def deprecated(
    removal_version: str,
    alternative: str | None = None,
    extra_message: str | None = None,
) -> Callable[[F], F]:
    """Mark a function or class as deprecated.

    Parameters
    ----------
    removal_version : str
        Version in which the decorated feature will be removed.
    alternative : str, optional
        Recommended alternative to use instead.
    extra_message : str, optional
        Additional context or migration instructions.

    Returns
    -------
    Callable
        Decorator function that wraps the target function or class.

    Examples
    --------
    >>> @deprecated("1.0.0", "new_function")
    ... def old_function():
    ...     pass

    >>> class OldClass:
    ...     @deprecated("1.0.0", "NewClass")
    ...     def old_method(self):
    ...         pass
    """

    def decorator(func_or_class: F) -> F:
        feature_name = f"{func_or_class.__module__}.{func_or_class.__qualname__}"

        if isinstance(func_or_class, type):
            # Handle class deprecation
            original_init = func_or_class.__init__

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                DeprecationHelper.warn(
                    feature_name,
                    removal_version,
                    alternative,
                    extra_message,
                )
                original_init(self, *args, **kwargs)

            func_or_class.__init__ = new_init
        else:
            # Handle function deprecation
            @functools.wraps(func_or_class)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                DeprecationHelper.warn(
                    feature_name,
                    removal_version,
                    alternative,
                    extra_message,
                )
                return func_or_class(*args, **kwargs)

            return wrapper  # type: ignore

        return func_or_class  # type: ignore[return-value]

    return decorator


def warn_deprecated(
    feature_name: str,
    removal_version: str,
    alternative: str | None = None,
    extra_message: str | None = None,
) -> None:
    """Issue a deprecation warning.

    Parameters
    ----------
    feature_name : str
        Name of the deprecated feature.
    removal_version : str
        Version in which the feature will be removed.
    alternative : str, optional
        Recommended alternative to use instead.
    extra_message : str, optional
        Additional context or migration instructions.

    Examples
    --------
    >>> warn_deprecated("old_config_key", "1.0.0", "new_config_key")
    """
    DeprecationHelper.warn(feature_name, removal_version, alternative, extra_message)
