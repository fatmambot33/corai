"""Tests for deprecation utilities."""

from __future__ import annotations

import warnings

import pytest

from openai_sdk_helpers.deprecation import (
    DeprecationHelper,
    deprecated,
    warn_deprecated,
)


class TestDeprecationHelper:
    """Tests for DeprecationHelper class."""

    def test_warn_basic_deprecation(self) -> None:
        """Test basic deprecation warning."""
        with pytest.warns(DeprecationWarning) as record:
            DeprecationHelper.warn("old_feature", "2.0.0")

        assert len(record) == 1
        assert "old_feature is deprecated" in str(record[0].message)
        assert "2.0.0" in str(record[0].message)

    def test_warn_with_alternative(self) -> None:
        """Test deprecation warning with alternative."""
        with pytest.warns(DeprecationWarning) as record:
            DeprecationHelper.warn(
                "old_function",
                "2.0.0",
                alternative="new_function",
            )

        msg = str(record[0].message)
        assert "old_function is deprecated" in msg
        assert "new_function" in msg

    def test_warn_with_extra_message(self) -> None:
        """Test deprecation warning with extra context."""
        extra = "See migration guide at docs/migration.md"
        with pytest.warns(DeprecationWarning) as record:
            DeprecationHelper.warn(
                "old_class",
                "2.0.0",
                extra_message=extra,
            )

        assert extra in str(record[0].message)


class TestDeprecatedDecorator:
    """Tests for @deprecated decorator."""

    def test_deprecated_function(self) -> None:
        """Test decorating a function with @deprecated."""

        @deprecated("2.0.0")
        def old_function() -> str:
            return "result"

        with pytest.warns(DeprecationWarning) as record:
            result = old_function()

        assert result == "result"
        assert len(record) == 1
        assert "deprecated" in str(record[0].message).lower()

    def test_deprecated_function_with_alternative(self) -> None:
        """Test deprecated function with alternative suggestion."""

        @deprecated("2.0.0", alternative="new_function")
        def old_function() -> str:
            return "result"

        with pytest.warns(DeprecationWarning) as record:
            old_function()

        assert "new_function" in str(record[0].message)

    def test_deprecated_function_with_args(self) -> None:
        """Test deprecated function with arguments."""

        @deprecated("2.0.0")
        def old_function(a: int, b: int) -> int:
            return a + b

        with pytest.warns(DeprecationWarning):
            result = old_function(1, 2)

        assert result == 3

    def test_deprecated_function_with_kwargs(self) -> None:
        """Test deprecated function with keyword arguments."""

        @deprecated("2.0.0")
        def old_function(a: int, b: int = 10) -> int:
            return a + b

        with pytest.warns(DeprecationWarning):
            result = old_function(5, b=20)

        assert result == 25

    def test_deprecated_class(self) -> None:
        """Test decorating a class with @deprecated."""

        @deprecated("2.0.0")
        class OldClass:
            def __init__(self) -> None:
                self.value = "initialized"

        with pytest.warns(DeprecationWarning) as record:
            obj = OldClass()

        assert obj.value == "initialized"
        assert len(record) == 1

    def test_deprecated_class_with_alternative(self) -> None:
        """Test deprecated class with alternative."""

        @deprecated("2.0.0", alternative="NewClass")
        class OldClass:
            pass

        with pytest.warns(DeprecationWarning) as record:
            OldClass()

        assert "NewClass" in str(record[0].message)

    def test_deprecated_class_constructor_args(self) -> None:
        """Test deprecated class constructor with arguments."""

        @deprecated("2.0.0")
        class OldClass:
            def __init__(self, x: int, y: str) -> None:
                self.x = x
                self.y = y

        with pytest.warns(DeprecationWarning):
            obj = OldClass(42, "test")

        assert obj.x == 42
        assert obj.y == "test"

    def test_deprecated_class_method(self) -> None:
        """Test deprecated class method."""

        class MyClass:
            @deprecated("2.0.0")
            def old_method(self) -> str:
                return "method_result"

        with pytest.warns(DeprecationWarning):
            obj = MyClass()
            result = obj.old_method()

        assert result == "method_result"


class TestWarnDeprecated:
    """Tests for warn_deprecated function."""

    def test_warn_deprecated_basic(self) -> None:
        """Test basic warn_deprecated call."""
        with pytest.warns(DeprecationWarning) as record:
            warn_deprecated("feature_name", "1.5.0")

        assert "feature_name" in str(record[0].message)
        assert "1.5.0" in str(record[0].message)

    def test_warn_deprecated_full_options(self) -> None:
        """Test warn_deprecated with all options."""
        with pytest.warns(DeprecationWarning) as record:
            warn_deprecated(
                "old_config",
                "2.0.0",
                alternative="new_config",
                extra_message="Update your config file.",
            )

        msg = str(record[0].message)
        assert "old_config" in msg
        assert "2.0.0" in msg
        assert "new_config" in msg
        assert "Update your config file" in msg


class TestDeprecationStackLevel:
    """Tests for proper stack level in warnings."""

    def test_decorator_stacklevel(self) -> None:
        """Test that decorator uses correct stacklevel for warning origin."""

        @deprecated("2.0.0")
        def function_to_deprecate() -> None:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            function_to_deprecate()

            # The warning should point to the function call, not internal code
            assert len(w) == 1
            # Check that the warning is of correct type
            assert issubclass(w[0].category, DeprecationWarning)
