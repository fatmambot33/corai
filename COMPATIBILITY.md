# SDK Compatibility Matrix

This document tracks tested and supported versions of the OpenAI SDKs.

## Supported SDK Versions

| openai-sdk-helpers | openai  | openai-agents | Python | Status |
|-------------------|---------|---------------|--------|--------|
| 0.1.x             | ≥1.0    | ≥0.1          | 3.10+  | Active |

## SDK Version Details

### OpenAI SDK (`openai`)

The `openai` SDK is used by the `response` module for direct API interactions.

**Minimum Version**: 1.0.0
**Tested Version**: 2.14.0
**Breaking Changes**: None documented

### OpenAI Agents SDK (`openai-agents`)

The `openai-agents` SDK is used by the `agent` module for high-level agent workflows.

**Minimum Version**: 0.1.0
**Tested Version**: 0.6.4
**Breaking Changes**: API is in beta, expect changes

## Version Constraints

Current constraints in `pyproject.toml`:

```toml
dependencies = [
    "openai>=2.14.0,<3.0.0",
    "openai-agents>=0.6.4,<1.0.0",
]
```

## Testing Strategy

- CI runs tests against multiple Python versions (3.10, 3.11, 3.12, 3.13)
- SDK versions are tested with latest stable releases
- Breaking changes are documented in release notes

## Known Issues

### OpenAI SDK

- No known compatibility issues

### OpenAI Agents SDK

- API is evolving; expect breaking changes in minor versions
- Monitor: https://github.com/openai/openai-agents-python

## Migration Guides

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for SDK-specific usage patterns.

## Reporting Issues

If you encounter compatibility issues:

1. Check this document for known issues
2. Verify your SDK versions: `pip list | grep openai`
3. Open an issue with version details
