"""Shared AI helpers and base structures."""

from __future__ import annotations

from .environment import get_data_path, get_model
from .utils.async_utils import run_coroutine_thread_safe, run_coroutine_with_fallback
from .context_manager import (
    AsyncManagedResource,
    ManagedResource,
    async_context,
    ensure_closed,
    ensure_closed_async,
)
from .errors import (
    OpenAISDKError,
    ConfigurationError,
    PromptNotFoundError,
    AgentExecutionError,
    VectorStorageError,
    ToolExecutionError,
    ResponseGenerationError,
    InputValidationError,
    AsyncExecutionError,
    ResourceCleanupError,
)
from .retry import with_exponential_backoff
from .utils.validation import (
    validate_choice,
    validate_dict_mapping,
    validate_list_items,
    validate_max_length,
    validate_non_empty_string,
    validate_safe_path,
    validate_url_format,
)
from .structure import (
    StructureBase,
    SchemaOptions,
    PlanStructure,
    TaskStructure,
    WebSearchStructure,
    VectorSearchStructure,
    PromptStructure,
    spec_field,
    SummaryStructure,
    ExtendedSummaryStructure,
    ValidationResultStructure,
    AgentBlueprint,
    create_plan,
    execute_task,
    execute_plan,
)
from .prompt import PromptRenderer
from .config import OpenAISettings
from .files_api import FilesAPIManager, FilePurpose
from .vector_storage import VectorStorage, VectorStorageFileInfo, VectorStorageFileStats
from .agent import (
    AgentBase,
    AgentConfiguration,
    AgentEnum,
    CoordinatorAgent,
    SummarizerAgent,
    TranslatorAgent,
    ValidatorAgent,
    VectorSearch,
    WebAgentSearch,
)
from .response import (
    ResponseBase,
    ResponseMessage,
    ResponseMessages,
    ResponseToolCall,
    ResponseConfiguration,
    ResponseRegistry,
    get_default_registry,
    parse_tool_arguments,
    attach_vector_store,
)
from .tools import (
    serialize_tool_result,
    tool_handler_factory,
    StructureType,
    ToolSpec,
    build_tool_definitions,
)
from .config import build_openai_settings
from .utils.deprecation import (
    deprecated,
    warn_deprecated,
    DeprecationHelper,
)
from .utils.output_validation import (
    ValidationResult,
    ValidationRule,
    JSONSchemaValidator,
    SemanticValidator,
    LengthValidator,
    OutputValidator,
    validate_output,
)
from .types import (
    SupportsOpenAIClient,
    OpenAIClient,
)

__all__ = [
    # Environment utilities
    "get_data_path",
    "get_model",
    # Async utilities
    "run_coroutine_thread_safe",
    "run_coroutine_with_fallback",
    # Error classes
    "OpenAISDKError",
    "ConfigurationError",
    "PromptNotFoundError",
    "AgentExecutionError",
    "VectorStorageError",
    "ToolExecutionError",
    "ResponseGenerationError",
    "InputValidationError",
    "AsyncExecutionError",
    "ResourceCleanupError",
    # Retry utilities
    "with_exponential_backoff",
    # Context managers
    "ManagedResource",
    "AsyncManagedResource",
    "ensure_closed",
    "ensure_closed_async",
    "async_context",
    # Validation
    "validate_non_empty_string",
    "validate_max_length",
    "validate_url_format",
    "validate_dict_mapping",
    "validate_list_items",
    "validate_choice",
    "validate_safe_path",
    # Main structure classes
    "StructureBase",
    "SchemaOptions",
    "spec_field",
    "PromptRenderer",
    "OpenAISettings",
    "FilesAPIManager",
    "FilePurpose",
    "VectorStorage",
    "VectorStorageFileInfo",
    "VectorStorageFileStats",
    "SummaryStructure",
    "PromptStructure",
    "AgentBlueprint",
    "TaskStructure",
    "PlanStructure",
    "AgentEnum",
    "AgentBase",
    "AgentConfiguration",
    "CoordinatorAgent",
    "SummarizerAgent",
    "TranslatorAgent",
    "ValidatorAgent",
    "VectorSearch",
    "WebAgentSearch",
    "ExtendedSummaryStructure",
    "WebSearchStructure",
    "VectorSearchStructure",
    "ValidationResultStructure",
    "ResponseBase",
    "ResponseMessage",
    "ResponseMessages",
    "ResponseToolCall",
    "ResponseConfiguration",
    "ResponseRegistry",
    "get_default_registry",
    "parse_tool_arguments",
    "attach_vector_store",
    "serialize_tool_result",
    "tool_handler_factory",
    "StructureType",
    "ToolSpec",
    "build_tool_definitions",
    "build_openai_settings",
    "create_plan",
    "execute_task",
    "execute_plan",
    # Type definitions
    "SupportsOpenAIClient",
    "OpenAIClient",
    # Deprecation utilities
    "deprecated",
    "warn_deprecated",
    "DeprecationHelper",
    # Output validation
    "ValidationResult",
    "ValidationRule",
    "JSONSchemaValidator",
    "SemanticValidator",
    "LengthValidator",
    "OutputValidator",
    "validate_output",
]
