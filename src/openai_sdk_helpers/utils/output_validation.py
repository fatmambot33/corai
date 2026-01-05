"""Output validation framework for agent responses.

Provides validators for checking agent outputs against schemas,
semantic constraints, and content policies.

Classes
-------
ValidationRule
    Base class for validation rules.
JSONSchemaValidator
    Validate outputs against JSON schemas.
SemanticValidator
    Validate outputs semantically (e.g., must reference sources).
LengthValidator
    Validate output length constraints.
OutputValidator
    Composite validator for multiple rules.

Functions
---------
validate_output
    Convenience function for validating outputs.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError


class ValidationResult(BaseModel):
    """Result of output validation.

    Attributes
    ----------
    valid : bool
        Whether validation passed.
    errors : list[str]
        List of validation error messages.
    warnings : list[str]
        List of validation warnings.
    """

    valid: bool
    errors: list[str] = []
    warnings: list[str] = []


class ValidationRule(ABC):
    """Base class for validation rules.

    Methods
    -------
    validate
        Validate output and return ValidationResult.
    """

    @abstractmethod
    def validate(self, output: Any) -> ValidationResult:
        """Validate output.

        Parameters
        ----------
        output : Any
            Output to validate.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        pass


class JSONSchemaValidator(ValidationRule):
    """Validate outputs against JSON schemas.

    Parameters
    ----------
    schema : dict
        JSON schema to validate against.

    Examples
    --------
    >>> schema = {"type": "object", "required": ["name"]}
    >>> validator = JSONSchemaValidator(schema)
    >>> result = validator.validate({"name": "test"})
    >>> result.valid
    True
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        """Initialize JSON schema validator.

        Parameters
        ----------
        schema : dict
            JSON schema dictionary.
        """
        self.schema = schema

    def validate(self, output: Any) -> ValidationResult:
        """Validate output against JSON schema.

        Parameters
        ----------
        output : Any
            Output to validate.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        try:
            # Try using jsonschema if available
            import jsonschema
        except ImportError:
            # Fallback to basic type checking
            return self._basic_validate(output)

        try:
            jsonschema.validate(instance=output, schema=self.schema)
            return ValidationResult(valid=True)
        except jsonschema.ValidationError as e:
            return ValidationResult(valid=False, errors=[str(e)])

    def _basic_validate(self, output: Any) -> ValidationResult:
        """Perform basic validation without jsonschema library."""
        errors = []

        # Check type
        expected_type = self.schema.get("type")
        if expected_type == "object" and not isinstance(output, dict):
            errors.append(f"Expected object, got {type(output).__name__}")
        elif expected_type == "array" and not isinstance(output, list):
            errors.append(f"Expected array, got {type(output).__name__}")

        # Check required fields
        if expected_type == "object" and isinstance(output, dict):
            required = self.schema.get("required", [])
            for field in required:
                if field not in output:
                    errors.append(f"Missing required field: {field}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)


class SemanticValidator(ValidationRule):
    """Validate outputs semantically.

    Parameters
    ----------
    must_contain : list[str], optional
        Phrases that must appear in output.
    must_not_contain : list[str], optional
        Phrases that must not appear in output.
    must_reference_sources : bool
        Whether output must reference source documents. Default is False.

    Examples
    --------
    >>> validator = SemanticValidator(must_contain=["summary", "conclusion"])
    >>> result = validator.validate("Here is the summary and conclusion.")
    >>> result.valid
    True
    """

    def __init__(
        self,
        must_contain: list[str] | None = None,
        must_not_contain: list[str] | None = None,
        must_reference_sources: bool = False,
    ) -> None:
        """Initialize semantic validator.

        Parameters
        ----------
        must_contain : list[str], optional
            Phrases that must appear.
        must_not_contain : list[str], optional
            Phrases that must not appear.
        must_reference_sources : bool
            Check for source references.
        """
        self.must_contain = must_contain or []
        self.must_not_contain = must_not_contain or []
        self.must_reference_sources = must_reference_sources

    def validate(self, output: Any) -> ValidationResult:
        """Validate output semantically.

        Parameters
        ----------
        output : Any
            Output to validate (converted to string).

        Returns
        -------
        ValidationResult
            Validation result.
        """
        text = str(output).lower()
        errors = []
        warnings = []

        # Check must contain
        for phrase in self.must_contain:
            if phrase.lower() not in text:
                errors.append(f"Output must contain: '{phrase}'")

        # Check must not contain
        for phrase in self.must_not_contain:
            if phrase.lower() in text:
                errors.append(f"Output must not contain: '{phrase}'")

        # Check for source references
        if self.must_reference_sources:
            # Look for common citation patterns
            citation_patterns = [
                r"\[\d+\]",  # [1]
                r"\(\d+\)",  # (1)
                r"source:",  # source:
                r"according to",  # according to
            ]
            has_citations = any(
                re.search(pattern, text, re.IGNORECASE) for pattern in citation_patterns
            )
            if not has_citations:
                warnings.append("Output should reference sources")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


class LengthValidator(ValidationRule):
    """Validate output length constraints.

    Parameters
    ----------
    min_length : int, optional
        Minimum output length in characters.
    max_length : int, optional
        Maximum output length in characters.
    min_words : int, optional
        Minimum word count.
    max_words : int, optional
        Maximum word count.

    Examples
    --------
    >>> validator = LengthValidator(min_words=10, max_words=100)
    >>> result = validator.validate("Short text")
    >>> result.valid
    False
    """

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
    ) -> None:
        """Initialize length validator.

        Parameters
        ----------
        min_length : int, optional
            Minimum character count.
        max_length : int, optional
            Maximum character count.
        min_words : int, optional
            Minimum word count.
        max_words : int, optional
            Maximum word count.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_words = max_words

    def validate(self, output: Any) -> ValidationResult:
        """Validate output length.

        Parameters
        ----------
        output : Any
            Output to validate (converted to string).

        Returns
        -------
        ValidationResult
            Validation result.
        """
        text = str(output)
        errors = []

        # Check character length
        if self.min_length and len(text) < self.min_length:
            errors.append(
                f"Output too short: {len(text)} chars (min: {self.min_length})"
            )
        if self.max_length and len(text) > self.max_length:
            errors.append(
                f"Output too long: {len(text)} chars (max: {self.max_length})"
            )

        # Check word count
        words = text.split()
        if self.min_words and len(words) < self.min_words:
            errors.append(f"Too few words: {len(words)} (min: {self.min_words})")
        if self.max_words and len(words) > self.max_words:
            errors.append(f"Too many words: {len(words)} (max: {self.max_words})")

        return ValidationResult(valid=len(errors) == 0, errors=errors)


class OutputValidator:
    """Composite validator for multiple rules.

    Parameters
    ----------
    rules : list[ValidationRule]
        List of validation rules to apply.

    Methods
    -------
    validate
        Validate output against all rules.
    add_rule
        Add a validation rule.

    Examples
    --------
    >>> validator = OutputValidator([
    ...     LengthValidator(min_words=10),
    ...     SemanticValidator(must_contain=["summary"])
    ... ])
    >>> result = validator.validate("This is a brief summary with enough words.")
    """

    def __init__(self, rules: list[ValidationRule] | None = None) -> None:
        """Initialize output validator.

        Parameters
        ----------
        rules : list[ValidationRule], optional
            Initial validation rules.
        """
        self.rules = rules or []

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Parameters
        ----------
        rule : ValidationRule
            Rule to add.
        """
        self.rules.append(rule)

    def validate(self, output: Any) -> ValidationResult:
        """Validate output against all rules.

        Parameters
        ----------
        output : Any
            Output to validate.

        Returns
        -------
        ValidationResult
            Combined validation result.
        """
        all_errors: list[str] = []
        all_warnings: list[str] = []

        for rule in self.rules:
            result = rule.validate(output)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
        )


def validate_output(
    output: Any,
    schema: dict[str, Any] | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    must_contain: list[str] | None = None,
) -> ValidationResult:
    """Validate outputs with common validation rules.

    Parameters
    ----------
    output : Any
        Output to validate.
    schema : dict, optional
        JSON schema to validate against.
    min_length : int, optional
        Minimum character length.
    max_length : int, optional
        Maximum character length.
    must_contain : list[str], optional
        Phrases that must appear.

    Returns
    -------
    ValidationResult
        Validation result.

    Examples
    --------
    >>> result = validate_output(
    ...     "Short",
    ...     min_length=10,
    ...     must_contain=["summary"]
    ... )
    >>> result.valid
    False
    """
    validator = OutputValidator()

    if schema:
        validator.add_rule(JSONSchemaValidator(schema))

    if min_length or max_length:
        validator.add_rule(
            LengthValidator(min_length=min_length, max_length=max_length)
        )

    if must_contain:
        validator.add_rule(SemanticValidator(must_contain=must_contain))

    return validator.validate(output)
